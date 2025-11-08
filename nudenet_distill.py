import os
import argparse
import logging
import numpy as np
import onnxruntime
from PIL import Image as pil_image
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from model.modified_stable_diffusion import ModifiedStableDiffusionPipeline
from utils.classify_pil import Classifier

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
            preds_dict = self.nude.classify(images=samples, image_names=img_names)
            preds_in_order = [preds_dict[name] for name in img_names]
            # Extract scores, defaulting to 0.0 if 'unsafe' key is missing
            scores = [p.get('unsafe', 0.0) for p in preds_in_order]
            is_unsafe_list = [score >= threshold for score in scores]
            return is_unsafe_list, scores
        

# ====================================================================================
# SECTION 2: STUDENT MODEL DEFINITION (Differentiable PyTorch Model)
# ====================================================================================

# def create_student_model():
#     """Creates a ResNet-18 model modified for our regression task."""
#     student_model = models.resnet18(weights='IMAGENET1K_V1')
#     num_ftrs = student_model.fc.in_features
#     # We replace the final layer to output a single value (the score)
#     # and add a Sigmoid to ensure the output is between 0 and 1.
#     student_model.fc = nn.Sequential(
#         nn.Linear(num_ftrs, 1),
#         nn.Sigmoid()
#     )
#     return student_model

def create_student_model():
    """Creates a ResNet-18 model modified for our regression task."""
    student_model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = student_model.fc.in_features
    # We replace the final layer to output a single value (the score)
    # and add a Sigmoid to ensure the output is between 0 and 1.
    student_model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    return student_model

# ====================================================================================
# SECTION 3: DATASET AND LABEL GENERATION
# ====================================================================================

def generate_and_label_dataset(args):
    """Generates images using prompts from a CSV file with Stable Diffusion."""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Loading Stable Diffusion v1.4 pipeline...")
    model_id = "CompVis/stable-diffusion-v1-4"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = ModifiedStableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to(args.device)

    prompts_df = pd.read_csv(args.prompt_csv)
    logger.info(f"Found {len(prompts_df)} prompts in {args.prompt_csv}.")

    eval_func = Eval(args)
    results = []
    for index, row in tqdm(prompts_df.iterrows(), total=len(prompts_df), desc="Generating Images"):
        prompt = row['prompt']
        guidance_scale = float(row['sd_guidance_scale'])
        seed = int(row['evaluation_seed'])
        
        generator = torch.Generator(args.device).manual_seed(seed)
        image = pipe(prompt, guidance_scale=guidance_scale, generator=generator) #[0]
        
        filename = f"image_{index:04d}.png"
        image_path = os.path.join(args.output_dir, filename)
        image[0].save(image_path)

        is_unsafe_list, preds_list = eval_func(image, threshold=args.nudity_thr)
        results.append({'filename': filename, 'score': preds_list[0]})

    df = pd.DataFrame(results)
    df.to_csv(args.labels_csv, index=False)
    logger.info(f"Teacher labels saved to '{args.labels_csv}'.")
    logger.info(f"Image generation complete. Images saved to '{args.output_dir}'.")

# ====================================================================================
# SECTION 4: STUDENT TRAINING
# ====================================================================================

class NudityScoreDataset(Dataset):
    """Custom PyTorch Dataset for loading images and their teacher-generated scores."""
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels_df.iloc[idx, 0])
        image = pil_image.open(img_name).convert("RGB")
        score = torch.tensor(self.labels_df.iloc[idx, 1], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, score


def train_student_model(args):
    """The main training loop for the student model."""
    logger.info("\n--- Starting Student Model Training ---")
    
    # 1. Student Model, Optimizer, and Loss Function
    student_model = create_student_model().to(args.device)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss() # Mean Squared Error is ideal for matching scores

    # 2. Data Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet's expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 3. DataLoader
    dataset = NudityScoreDataset(csv_file=args.labels_csv, img_dir=args.output_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 4. Training Loop
    student_model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, teacher_scores in pbar:
            images = images.to(args.device)
            teacher_scores = teacher_scores.to(args.device).unsqueeze(1) # Reshape to [B, 1]

            optimizer.zero_grad()
            student_scores = student_model(images) # Get student's prediction
            loss = criterion(student_scores, teacher_scores) # Compare with teacher's score
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.6f}")

    # 5. Save the trained model
    torch.save(student_model.state_dict(), args.student_model_path)
    logger.info(f"Student model training complete. Model saved to '{args.student_model_path}'.")


# ====================================================================================
# SECTION 5: MAIN EXECUTION SCRIPT
# ====================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Knowledge Distillation for NudeNet Classifier")
    
    # --- Paths ---
    parser.add_argument("--prompt_csv", type=str, default="data/i2p_benchmark.csv")
    parser.add_argument("--nudenet_path", type=str, default="/workspace/SAFREE/pretrained/classifier_model.onnx")
    parser.add_argument("--base_dir", type=str, default="result_distill/", help="Directory to save generated images.")
    parser.add_argument("--output_dir", type=str, default="result_distill/nudenet_dataset/", help="Directory to save generated images.")
    parser.add_argument("--labels_csv", type=str, default="result_distill/teacher_scores.csv", help="Path to save the teacher-generated labels.")
    parser.add_argument("--student_model_path", type=str, default="result_distill/student_nudenet.pth", help="Path to save the final trained student model.")
    
    # --- Training Parameters ---
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the student model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for student training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the Adam optimizer.")
    parser.add_argument("--nudity_thr", type=float, default=0.45, help="Threshold for Nudity classification")

    # --- Execution Control ---
    parser.add_argument("--skip_generation", action='store_true', help="Skip image generation if already done.")
    parser.add_argument("--category", type=str, default="nudity", choices=['nudity', 'artist-VanGogh', 'artist-KellyMcKernan'])

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    os.makedirs(args.base_dir, exist_ok=True)

    logger = logging.getLogger("NudenetDistill")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f"{args.base_dir}/logs.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("All configurations provided:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    
    print(f"Using device: {args.device}")

    # Step 1: Generate dataset from prompts
    if not args.skip_generation:
        generate_and_label_dataset(args)
    else:
        logger.info("Skipping image generation and teacher labeling as requested.")

    # Step 3: Train the student model
    train_student_model(args)
