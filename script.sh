python train_neg_emb_avg1_rep.py \
    --data ./data/nudity_safe_neg_subset.csv \
    --output_dir ./results/neg_emb_ddim_learn1avg_sub_b64_4_i10_l1e2_out-r/ \
    --model_id CompVis/stable-diffusion-v1-4 \
    --lr 1e-2 \
    --batch_size 64 \
    --micro_batch_size 4 \
    --num_inner_steps 10 \
    --guidance_scale 7.5 \
    --nudity_thr 0.45 \
    --device cuda \
    --num_ddim_steps 50


# python baseline.py \
#     --data ./data/nudity_safe_neg_s_remove.csv \
#     --output_dir ./results/original_ddim/ \
#     --model_id CompVis/stable-diffusion-v1-4 \
#     --guidance_scale 7.5 \
#     --nudity_thr 0.45

# python train_neg_emb_avg1.py \
#     --data ./data/nudity_safe_neg_subset.csv \
#     --output_dir ./results/neg_emb_ddim_selected_learn1avg-subtry/ \
#     --model_id CompVis/stable-diffusion-v1-4 \
#     --batch_size 4 \
#     --guidance_scale 7.5 \
#     --nudity_thr 0.45 \
#     --device cuda \
#     --num_ddim_steps 50

# python train_neg_emb_avg1_same.py \
#     --data ./data/nudity_safe_neg_selected.csv \
#     --output_dir ./results/neg_emb_ddim_selected_learn1avg_samep_bs10/ \
#     --model_id CompVis/stable-diffusion-v1-4 \
#     --batch_size 10 \
#     --guidance_scale 7.5 \
#     --nudity_thr 0.45 \
#     --device cuda \
#     --num_ddim_steps 50