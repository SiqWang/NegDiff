# python nudenet_distill.py \
#     --prompt_csv ./data/i2p_benchmark.csv \
#     --base_dir ./result_distill/ \
#     --output_dir ./result_distill/nudenet_dataset/ \
#     --labels_csv ./result_distill/teacher_scores.csv \
#     --student_model_path ./result_distill/student_nudenet.pth \
#     --nudity_thr 0.45 \
#     --epochs 50 \
#     --batch_size 16 \
#     --learning_rate 1e-4 \

deepspeed --num_gpus=2 train_prompt_emb_classifier_dszero3.py --deepspeed ds_zero3.json \
    --dataset_path ./data/nudity_safe_neg_subset.csv \
    --output_dir ./results_prompt/prompt_emb_ddim_sub_b41_in10_lr1e3_zero3/ \
    --model_id CompVis/stable-diffusion-v1-4 \
    --student_model_path ./result_distill/student_nudenet.pth \
    --lr 1e-3 \
    --batch_size 1 \
    --micro_batch_size 1 \
    --num_inner_steps 1 \
    --guidance_scale 7.5 \
    --nudity_thr 0.45 \
    --device cuda \
    --num_ddim_steps 50

# python train_prompt_emb_classifier_loss.py \
#     --data ./data/nudity_safe_neg_subset.csv \
#     --output_dir ./results_prompt/prompt_emb_ddim_sub_b41_in10_lr1e4ori/ \
#     --model_id CompVis/stable-diffusion-v1-4 \
#     --student_model_path ./result_distill/student_nudenet.pth \
#     --lr 1e-4 \
#     --batch_size 4 \
#     --micro_batch_size 1 \
#     --num_inner_steps 10 \
#     --guidance_scale 7.5 \
#     --nudity_thr 0.45 \
#     --device cuda:1 \
#     --num_ddim_steps 50

# python train_neg_emb_avg1_same_rep.py \
#     --data ./data/nudity_safe_neg_subset.csv \
#     --output_dir ./results/neg_emb_ddim_learn1avg_samep_sub_b32_i50_l5e2_out_n-r/ \
#     --model_id CompVis/stable-diffusion-v1-4 \
#     --lr 5e-2 \
#     --batch_size 32 \
#     --micro_batch_size 1 \
#     --num_inner_steps 50 \
#     --guidance_scale 7.5 \
#     --nudity_thr 0.45 \
#     --device cuda \
#     --num_ddim_steps 50


# torchrun --nproc_per_node=4 train_neg_emb_multi.py \
#     --data /workspace/Neg_Null/data/split_data/nudity_train_data.csv \
#     --output_dir ./results/neg_emb_ddim_split_train_4_f/ \
#     --model_id CompVis/stable-diffusion-v1-4 \
#     --guidance_scale 7.5 \
#     --nudity_thr 0.45