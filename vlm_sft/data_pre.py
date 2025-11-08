import os
import re
import json
from glob import glob

log_path = "/workspace/NegDiff/results_pre/SD_only-emp_v1-4_mma-diffusion/logs.txt"
img_dir = "mma_data"
output_path = "mma_data.jsonl"

# # Regular expressions to extract fields
# pattern_prompt = re.compile(r".*Case#:\s*(\d+):\s*target prompt:\s*(.*)")
# pattern_result = re.compile(r".*Optimized image is unsafe:\s*(\w+),\s*toxicity pred:\s*([\d.]+)")

# dataset = []
# with open(log_path, "r", encoding="utf-8") as f:
#     lines = f.readlines()

# # Iterate over every two lines (prompt + result)
# for i in range(1, len(lines), 2):
#     line1 = lines[i]
#     line2 = lines[i + 1]

#     match_prompt = pattern_prompt.search(line1)
#     match_result = pattern_result.search(line2)

#     if not (match_prompt and match_result):
#         continue

#     case_num = match_prompt.group(1)
#     prompt = match_prompt.group(2).strip()

#     unsafe_flag = match_result.group(1) == "True"
#     toxicity = float(match_result.group(2))

#     # Find image file with that case number
#     img_candidates = glob(os.path.join(img_dir, f"{case_num}_*.png"))
#     image_path = img_candidates[0] if img_candidates else None

#     if not image_path:
#         print(f"⚠️ Missing image for case {case_num}")
#         continue

#     label = "unsafe" if unsafe_flag else "safe"

#     sample = {
#         "instruction": "Judge whether this image is safe. Output safe / unsafe.",
#         # "input": f"Prompt: {prompt}",
#         "image": image_path,
#         "label": label,
#         "score": round(toxicity, 3)
#     }
#     dataset.append(sample)

# # Write JSONL file
# with open(output_path, "w", encoding="utf-8") as f:
#     for item in dataset:
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")

# print(f"✅ Dataset saved to {output_path}, total samples: {len(dataset)}")


# data_list = []

# with open(output_path, "r", encoding="utf-8") as infile:
#     for line in infile:
#         data = json.loads(line.strip())

#         instruction = data.get("instruction", "").strip()
#         image_path = data.get("image", "").strip()
#         text_input = data.get("input", "").strip()
#         label = data.get("label", "")
#         score = data.get("score", None)

#         new_record = {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": f"<image>{instruction}" # Here is the prompt: {text_input}"
#                 },
#                 {
#                     "role": "assistant",
#                     "content": f"Label: {label} | Score: {score}"
#                 }
#             ],
#             "images": [image_path]
#         }

#         data_list.append(new_record)

# output_path = "mma_chat.json"
# with open(output_path, "w", encoding="utf-8") as outfile:
#     json.dump(data_list, outfile, ensure_ascii=False, indent=2)

# print(f"✅ Converted data saved to: {output_path}")



# input_path = output_path
# output_path = "sft_dataset_pretty.json"

# data = []
# with open(input_path, "r", encoding="utf-8") as f:
#     for line in f:
#         data.append(json.loads(line))

# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)

# print(f"✅ Pretty JSON saved to {output_path}")



import json
import random

# Parameters
input_file = "mma_chat.json"   # your input .jsonl file
train_output = "mma_train.json"
eval_output = "mma_eval.json"
split_ratio = 0.9

# Read JSONL file
with open(input_file, "r", encoding="utf-8") as f:
    # data = [json.loads(line) for line in f]   # for jsonl file
    data = json.load(f)  # for json file

# Shuffle data to randomize
random.seed(42)
random.shuffle(data)

# Split into train and eval sets
split_index = int(len(data) * split_ratio)
train_data = data[:split_index]
eval_data = data[split_index:]

# Save as JSON files
with open(train_output, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(eval_output, "w", encoding="utf-8") as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(train_data)} examples to {train_output}")
print(f"Saved {len(eval_data)} examples to {eval_output}")

