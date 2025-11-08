import os
import torch


base_dir = '/workspace/Neg_Null/results/neg_emb_ddim_selected_learn1avg_samep_2'

all_embeddings = []

for file in os.listdir(base_dir):
    if file.endswith('.pt'):
        pt_path = os.path.join(base_dir, file)
        emb_list = torch.load(pt_path)  # load tensor
        emb_list0 = [emb[:1] for emb in emb_list]
        all_embeddings.append(emb_list0)
        print(f'Loaded file {pt_path}')

num_files = len(all_embeddings)
num_timesteps = len(all_embeddings[0])
print(f"num_files: {num_files}, num_timesteps: {num_timesteps}")

avg_list = []

for t in range(num_timesteps):
    # collect the t-th tensor from each file
    tensors_t = [all_embeddings[f][t] for f in range(num_files)]
    # stack and average
    avg_t = torch.stack(tensors_t, dim=0).mean(dim=0)
    avg_list.append(avg_t)  # shape (1,77,768)

# save the list of 50 averaged tensors
save_path = os.path.join(base_dir, "avg_embedding.pt")
torch.save(avg_list, save_path)

print(f"Averaged embeddings saved to {save_path}, list length: {len(avg_list)}, each shape: {avg_list[0].shape}")
