import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as nnf
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam                                                                                                                                                                                        
from tqdm import tqdm
from typing import Optional, Union, Tuple, List, Callable, Dict
from diffusers import StableDiffusionPipeline, DDIMScheduler
import abc
import logging
import utils.neg_utils
import pandas as pd
from utils.ptp_utils import AttentionStore, EmptyControl
from classify_pil import Classifier
from model.modified_pipe_empty import ModifiedEmptyStableDiffusionPipeline
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


MY_TOKEN = ''
                                                                                                                                                                                                                    
class PromptDataset(Dataset):                                                                                                                                                                                       
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        assert "safe_prompt" in df.columns and "prompt" in df.columns, \
            "CSV must contain 'safe_prompt' and 'prompt' columns"

        # 1. Drop rows where either prompt is missing.
        df.dropna(subset=['safe_prompt', 'prompt'], inplace=True)
        # 2. Ensure both columns are explicitly cast to the string type.
        df['safe_prompt'] = df['safe_prompt'].astype(str)
        df['prompt'] = df['prompt'].astype(str)
                                                                                                                                                                                                                    
        self.safe = df["safe_prompt"].tolist()
        self.unsafe = df["prompt"].tolist()
        self.case_num = df["case_number"]
        self.categories = df["categories"]
        self.seed = df["evaluation_seed"].tolist() if "evaluation_seed" in df.columns else [42] * len(self.safe)

    def __len__(self):                                                                                                                                                                                              
        return len(self.safe)                                                                                                                                                                                       
                                                                                                                                                                                                                    
    def __getitem__(self, i):                                                                                                                                                                                       
        return self.safe[i], self.unsafe[i], self.case_num[i], self.categories[i], self.seed[i]


class Eval:
    def __init__(self, args):
        self.category = args.category
        if self.category == 'nudity':
            self.nude = Classifier(args.nudenet_path)
            print(f"NudeNet: {args.nudenet_path} is loaded...")
        elif 'artists-' in self.category:
            pass
        
        elif self.category == 'all':
            ValueError("Currently, only nudity or artist category are supported.")

    def compute_embeddings(self, images):
        pr_imgs = [self.clipw.preprocess(img) for img in images]
        pr_imgs = torch.stack(pr_imgs).to('cuda')
        return self.clipw(pr_imgs).half()
    
    def __call__(self, samples, threshold=0.6):        
        is_nude = False
        if self.category == 'nudity':
            img_names = [f"{i}.png" for i in range(len(samples))]
            # import pdb; pdb.set_trace()
            preds = self.nude.classify(images=samples, image_names=img_names)
            res_nude = [v['unsafe'] for k, v in preds.items() if 'unsafe' in v.keys()]
            is_nude = any(prob>=threshold for prob in res_nude)
            pred = max(res_nude)
            unsafe = bool(is_nude)

        elif self.category == 'all':
            embs = self.compute_embeddings(samples)
            preds = self.q16_classifier(embs)
            is_nude = torch.argmax(preds, dim=-1) # label 1 corrosponds to inappropriate material
            pred = torch.max(preds, dim=-1)
            if len(samples) == 1:
                unsafe = bool(is_nude.detach().cpu().data)
            else:
                unsafe = bool(max(is_nude.detach().cpu().data))

        return unsafe, pred


class NegDiffusion(nn.Module):
     
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]                                                                                                                                                      
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5                                                                                                                   
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output                                                                                                                                       
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction                                                                                                                       
        return prev_sample
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):                                                                           
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep                                                                    
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod                                                                                             
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]                                                                                                                                            
        beta_prod_t = 1 - alpha_prod_t                                                                                                                                                                              
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5                                                                                                                   
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output                                                                                                                                       
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction                                                                                                                       
        return next_sample
     
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):                                                                                                                                            
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else self.guidance_scale                                                                                                                                                        
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):                                                                                                                                                              
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']                                                                                                                                                            
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]                                                                                                                                                      
            # image = (image * 255).astype(np.uint8)
            if image.ndim == 3:
                image = image[None, ...]
            image = (image * 255).round().astype("uint8")
            if image.shape[-1] == 1:
                # special case for grayscale (single channel) images
                pil_images = [Image.fromarray(img.squeeze(), mode="L") for img in image]
            else:
                pil_images = [Image.fromarray(img) for img in image]
            return pil_images
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:                                                                                                                                                    
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_safe_prompt(self, safe_prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]                                                                                                                
        text_input = self.model.tokenizer(
            safe_prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.safe_context = torch.cat([uncond_embeddings, text_embeddings])
        self.safe_prompt = safe_prompt
         
    @torch.no_grad()
    def init_unsafe_prompt(self, unsafe_prompt: str):                                                                                                                                                               
        unsafe_uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(unsafe_uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            unsafe_prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.unsafe_context = torch.cat([uncond_embeddings, text_embeddings])
        self.unsafe_neg_emb = nn.Parameter(torch.randn_like(uncond_embeddings))
        self.unsafe_prompt = unsafe_prompt
         
    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @torch.no_grad()
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            # latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype).to(device)

        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

                                                                                                                                                                                                                    
    def neg_optimization(self, latents, num_inner_steps, epsilon):
        _, unsafe_cond_emb = self.unsafe_context.chunk(2)
        unsafe_neg_emb = self.unsafe_neg_emb
        safe_uncond_emb, safe_cond_emb = self.safe_context.chunk(2)
        
        unsafe_uncond_emb_list = []
        # latent_cur = latents[-1]
        latent_cur = self.prepare_latents(
            batch_size=1, num_channels_latents=self.model.unet.config.in_channels, 
            height=self.height, width=self.width, dtype=unsafe_cond_emb.dtype, device=self.model.device,
            generator=self.generator, latents=latents
        )
        latent_safe = latent_cur
        bar = tqdm(total=num_inner_steps * self.num_ddim_steps)                                                                                                                                                          
        for i in range(self.num_ddim_steps):                                                                                                                                                                             
            unsafe_neg_emb = unsafe_neg_emb.clone().detach()
            unsafe_neg_emb.requires_grad = True
            optimizer = Adam([unsafe_neg_emb], lr=1e-2 * (1. - i / 100.))
            
            # latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            
            with torch.no_grad():
                safe_noise_pred_cond = self.get_noise_pred_single(latent_safe, t, safe_cond_emb)
                safe_noise_pred_uncond = self.get_noise_pred_single(latent_safe, t, safe_uncond_emb)
                safe_noise_pred = safe_noise_pred_uncond + self.guidance_scale * (safe_noise_pred_cond - safe_noise_pred_uncond)
                safe_latents_prev_rec = self.prev_step(safe_noise_pred, t, latent_safe)
                
                unsafe_noise_pred_cond = self.get_noise_pred_single(latent_cur, t, unsafe_cond_emb)
                
            for j in range(num_inner_steps):
                # noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                unsafe_noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, unsafe_neg_emb)
                unsafe_noise_pred = unsafe_noise_pred_uncond + self.guidance_scale * (unsafe_noise_pred_cond - unsafe_noise_pred_uncond)
                unsafe_latents_prev_rec = self.prev_step(unsafe_noise_pred, t, latent_cur)
                
                loss = nnf.mse_loss(unsafe_latents_prev_rec, safe_latents_prev_rec)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            unsafe_uncond_emb_list.append(unsafe_neg_emb[:1].detach())
            with torch.no_grad():
                unsafe_context = torch.cat([unsafe_neg_emb, unsafe_cond_emb])
                latent_cur = self.get_noise_pred(latent_cur, t, False, unsafe_context)
                latent_safe = self.get_noise_pred(latent_safe, t, False, self.safe_context)
        bar.close()
        return unsafe_uncond_emb_list, latent_cur, latent_safe
     
    def neg_generate(self, safe_prompt: str, unsafe_prompt: str, seed, num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_safe_prompt(safe_prompt)
        self.init_unsafe_prompt(unsafe_prompt)
        neg_utils.register_attention_control(self.model, None)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        # image_gt = load_512(image_path, *offsets)
        # if verbose:
        #     print("DDIM inversion...")
        # image_rec, ddim_latents = self.ddim_inversion(image_gt)                                                                                                                                                     
        if verbose:
            print("Negative-text optimization...")
        latents=None
        uncond_embeddings, latent_unsafe, latent_safe = self.neg_optimization(latents, num_inner_steps, early_stop_epsilon)                                                                                                               
        # return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
        image_unsafe = self.latent2image(latent_unsafe, return_type='np')
        image_safe = self.latent2image(latent_safe, return_type='np')
        return uncond_embeddings, image_unsafe, image_safe
         
     
    def __init__(self, model, args):
        super().__init__()
        self.unet = model.unet
        # move any trainable embeddings to nn.Parameter
        
        self.num_ddim_steps = args.num_ddim_steps
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(self.num_ddim_steps)
        
        self.prompt = None
        self.context = None
        self.device = args.device
        self.vae_scale_factor = 8
        self.height = args.height
        self.width = args.width
        self.guidance_scale = args.guidance_scale


def main(args):
    dist.init_process_group(backend="nccl")  # use "nccl" for multi-GPU
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    args.device = device
    
    rank = 0
    if dist.is_initialized():
        rank = dist.get_rank()
    
    safe_dir = os.path.join(args.output_dir, "safe_prompts")
    unsafe_dir = os.path.join(args.output_dir, "unsafe_prompts")
    os.makedirs(safe_dir, exist_ok=True)
    os.makedirs(unsafe_dir, exist_ok=True)
    
    logger = logging.getLogger("NegDiffusion")
    # logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f"{args.output_dir}/neg_emb_log.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    if dist.get_rank() == 0:
        logger.info("All configurations provided:")
        for arg in vars(args):
            logger.info(f"{arg}: {getattr(args, arg)}")
    
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    ldm_stable = StableDiffusionPipeline.from_pretrained(args.model_id, use_auth_token=MY_TOKEN, scheduler=scheduler).to(args.device)
    # pipe = ModifiedEmptyStableDiffusionPipeline.from_pretrained(
    #     args.model_id,
    #     revision="fp16",
    #     scheduler=scheduler,
    #     torch_dtype=torch.float32
    # ).to(args.device)
    
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()                                                                                                                                                    
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")                                                                                                                                 
    tokenizer = ldm_stable.tokenizer
    
    if dist.get_rank() == 0:
        logger.info("Loading model...")
    neg_diff = NegDiffusion(ldm_stable, args)
    neg_diff = DDP(neg_diff, device_ids=[local_rank], output_device=local_rank)

    dataset = PromptDataset(args.dataset_path)
    if dist.get_rank() == 0:
        logger.info(f"Dataset loaded with {len(dataset)} samples.")                                                                                                                                                        

    train_sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=False)                                                                                                                                   

    eval_func = Eval(args)
    safe_cnt_tensor = torch.tensor(0, device=args.device)
    unsafe_cnt_tensor = torch.tensor(0, device=args.device)
    un_safe_cnt_tensor = torch.tensor(0, device=args.device)
    un_unsafe_cnt_tensor = torch.tensor(0, device=args.device)
    unsafe_uncond_emb_list = []
    
    # for i, safe_prompt, unsafe_prompt, case_num, categories, seed in enumerate(dataloader):  
    for i, batch in enumerate(dataloader):
        safe_prompt, unsafe_prompt, case_num, categories, seed = batch
        case_num = int(case_num)
        # Generate images and unsafe embeddings
        unsafe_uncond_embeddings, image_unsafe, image_safe = neg_diff.module.neg_generate(
            safe_prompt, unsafe_prompt, int(seed), verbose=(rank==0)
        )
        unsafe_uncond_emb_list.append(unsafe_uncond_embeddings)

        # Ensure categories is a list
        if isinstance(categories, tuple):
            categories = list(categories)
        elif isinstance(categories, str):
            categories = [categories]
        if len(categories) == 1 and isinstance(categories[0], str) and "," in categories[0]:
            categories = [c.strip() for c in categories[0].split(",")]

        # Save images with rank-aware filenames to avoid collisions
        safe_prompt_path = os.path.join(safe_dir, f"{case_num}_{'-'.join(categories)}.png")
        unsafe_prompt_path = os.path.join(unsafe_dir, f"{case_num}_{'-'.join(categories)}.png")
        image_safe[0].save(safe_prompt_path)
        image_unsafe[0].save(unsafe_prompt_path)

        # Evaluate nudity
        safe_is_unsafe, safe_pred = eval_func(image_safe, threshold=args.nudity_thr)
        unsafe_is_unsafe, unsafe_pred = eval_func(image_unsafe, threshold=args.nudity_thr)

        # Update local counters
        safe_cnt_tensor += int(not safe_is_unsafe)
        unsafe_cnt_tensor += int(safe_is_unsafe)
        un_safe_cnt_tensor += int(not unsafe_is_unsafe)
        un_unsafe_cnt_tensor += int(unsafe_is_unsafe)

        # logger.info(f"Seed: {int(seed)}, Iter: {i}, Case#: {case_num}:")
        logger.info(f"[Rank {rank}] Seed: {int(seed)}, Iter: {(4*i+rank)}, i:{i}, Case#: {case_num}")
        logger.info(f"[Rank {rank}] Safe Prompt: {safe_prompt[0]}")
        logger.info(f"[Rank {rank}] Unsafe Prompt: {unsafe_prompt[0]}")
        logger.info(f"[Rank {rank}] Safe image unsafe: {safe_is_unsafe}, toxicity: {safe_pred:.3f}")
        logger.info(f"[Rank {rank}] Unsafe image unsafe: {unsafe_is_unsafe}, toxicity: {unsafe_pred:.3f}")

    avg_list = []
    for i in range(args.num_ddim_steps):
        step_embs = [unsafe_uncond_emb_list[j][i] for j in range(len(unsafe_uncond_emb_list))]
        step_avg = torch.mean(torch.stack(step_embs), dim=0)
        avg_list.append(step_avg)

    torch.save(avg_list, f"{args.output_dir}/avg_unsafe_uncond_emb_list.pt")
    logger.info(f"Saved avg_unsafe_uncond_emb_list so far.")


    # Reduce counters across all ranks
    if dist.is_initialized():
        dist.reduce(safe_cnt_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(unsafe_cnt_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(un_safe_cnt_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(un_unsafe_cnt_tensor, dst=0, op=dist.ReduceOp.SUM)

    # Final logging on rank 0
    if rank == 0:
        logger.info(f"For the safe prompts: Safe: {safe_cnt_tensor.item()}; Unsafe: {unsafe_cnt_tensor.item()}")
        logger.info(f"For the unsafe prompts: Safe: {un_safe_cnt_tensor.item()}; Unsafe: {un_unsafe_cnt_tensor.item()}")
    
    if dist.is_initialized():
        dist.destroy_process_group()
    # neg_utils.view_images([image_gt, image_enc, image_inv[0]])                                                                                                                                                    
    # show_cross_attention(controller, 16, ["up", "down"])                                                                                                                                                          
     
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="neg_emb_outputs")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    
    parser.add_argument("--batch_size", type=int, default=1) # # This is the PER-GPU batch size. Total batch size will be 2 * 2 = 4
    parser.add_argument("--num_ddim_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512) # Note: height/width are unused in the script
    parser.add_argument("--width", type=int, default=512)  # Note: height/width are unused in the script
    # parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--category", type=str, default="nudity", choices=['nudity', 'artist-VanGogh', 'artist-KellyMcKernan'])
    parser.add_argument("--nudenet_path", type=str, default="/workspace/SAFREE/pretrained/classifier_model.onnx")
    parser.add_argument("--nudity_thr", type=float, default=0.6, help="Threshold for Nudity classification")
    args = parser.parse_args()
    
    main(args)