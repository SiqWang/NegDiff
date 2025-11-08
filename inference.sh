python inference.py \
    --model_id CompVis/stable-diffusion-v1-4 \
    --device cuda:1 \
    --avg_emb_path ./results/neg_emb_ddim_learn1avg_samep_sub_b16_i10_l1e2_out_n-r/unsafe_uncond_emb_iter_1_case_1011.pt \
    --harm_emb_path ./results_prompt/prompt_emb_ddim_sub_celoss_in10_lr1e2ori/unsafe_cond_emb_iter_14_case_1011.pt \
    --prompt_file ./data/nudity_safe_neg_selected_inf.csv \
    --seed 42 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --height 512 \
    --width 512 \
    --out_dir ./inference_results/neg_emb_ddim_selected_l1avg_same_b16_1011-harm-inf-o/

    # ./results/neg_emb_ddim_selected_learn1avg_samep_2/avg_embedding.pt
    # ./results/neg_emb_ddim_selected_learn1avg_samep_sub/avg_embedding.pt
    # ./results/neg_emb_ddim_learn1avg_samep_sub_b16_i10_l1e2_out_n-r/unsafe_uncond_emb_iter_1_case_1011.pt
    # ./results/neg_emb_ddim_learn1avg_sub_b64_4_i10_l1e2_out-r/unsafe_uncond_emb_iter_0.pt 