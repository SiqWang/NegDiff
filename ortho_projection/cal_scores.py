# pip install torch torchvision torcheval
# pip install transformers Pillow

import torch
from torchvision.transforms import ToTensor
import pandas as pd
from PIL import Image
import os
from torcheval.metrics.image import FrechetInceptionDistance
from transformers import CLIPProcessor, CLIPModel
import glob


def calculate_text_image_clip_scores(csv_path, image_folder_path):
    """
    Calculates CLIP scores between text prompts from a CSV and their corresponding images.

    Args:
        csv_path (str): The path to the CSV file containing prompts and case numbers.
        image_folder_path (str): The path to the folder containing the generated images.
    """

    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load CLIP Model ---
    print("Loading CLIP model...")
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("CLIP model loaded successfully.")
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return

    # --- Load CSV Data ---
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {csv_path} with {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    clip_scores = {}

    # --- Iterate through prompts and images ---
    for index, row in df.iterrows():
        try:
            prompt = row["prompt"]
            case_num = row["case_number"]

            # Find the corresponding image file. This handles filenames like '21_nudity.png', '21_violence.png', etc.
            # It will find the first file that starts with the case number.
            image_path_pattern = os.path.join(image_folder_path, f"{case_num}_*.png")
            image_files = glob.glob(image_path_pattern)

            if not image_files:
                print(f"Warning: No image found for case number {case_num} in {image_folder_path}")
                continue
            
            # Use the first match found
            image_path = image_files[0]
            image = Image.open(image_path).convert("RGB")

            # --- Calculate CLIP Score ---
            # The processor handles both text and image preprocessing
            inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True, truncation=True)

            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get the model outputs
            outputs = model(**inputs)
            
            # The logits_per_image gives the similarity score.
            # We scale it by 100 as is common practice with CLIP scores.
            clip_score = outputs.logits_per_image.item()
            
            clip_scores[case_num] = clip_score
            # print(f"Processed Case #{case_num}: CLIP Score = {clip_score:.4f}")

        except FileNotFoundError:
            print(f"Error: Image for case number {case_num} not found at expected path.")
        except Exception as e:
            print(f"An error occurred processing case number {case_num}: {e}")

    # --- Display Results ---
    print("\n--- Results ---")
    if not clip_scores:
        print("No CLIP scores were calculated.")
        return

    # Calculate and print the average CLIP score
    average_clip_score = sum(clip_scores.values()) / len(clip_scores)
    print(f"Average CLIP Score: {average_clip_score:.4f}")

    # print("\nIndividual CLIP Scores per Case:")
    # for case_num, score in clip_scores.items():
    #     print(f"  Case #{case_num}: {score:.4f}")


def calculate_scores(folder1_path, folder2_path):
    """
    Calculates the Frechet Inception Distance (FID) and CLIP scores for paired images in two folders.

    Args:
        folder1_path (str): The path to the first folder of images.
        folder2_path (str): The path to the second folder of images.
    """

    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load CLIP Model ---
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP model loaded.")

    # --- Initialize FID Metric ---
    fid_metric = FrechetInceptionDistance(device=device)

    # --- Get Image Files ---
    try:
        image_files_1 = sorted(os.listdir(folder1_path))
        image_files_2 = sorted(os.listdir(folder2_path))
    except FileNotFoundError:
        print("Error: One or both folders not found.")
        return

    if image_files_1 != image_files_2:
        print("Warning: The two folders do not contain the same image files.")
        # Find common files to proceed with calculation for the intersection of files
        common_files = sorted(list(set(image_files_1) & set(image_files_2)))
        if not common_files:
            print("Error: No common image files found in the two folders.")
            return
        print(f"Found {len(common_files)} common files. Proceeding with these.")
        image_files = common_files
    else:
        image_files = image_files_1


    # --- Process Images and Calculate Scores ---
    clip_scores = {}

    for image_name in image_files:
        image_path_1 = os.path.join(folder1_path, image_name)
        image_path_2 = os.path.join(folder2_path, image_name)

        try:
            # --- Load and Preprocess Images ---
            image1 = Image.open(image_path_1).convert("RGB")
            image2 = Image.open(image_path_2).convert("RGB")

            # --- FID Calculation ---
            # Convert images to float32 tensors in the range [0, 1] as expected by the metric.
            tensor1 = ToTensor()(image1)
            tensor2 = ToTensor()(image2)

            # Add a batch dimension
            tensor1 = tensor1.unsqueeze(0).to(device)
            tensor2 = tensor2.unsqueeze(0).to(device)

            # Update the FID metric
            fid_metric.update(tensor1, is_real=True)
            fid_metric.update(tensor2, is_real=False)


            # --- CLIP Score Calculation ---
            inputs = clip_processor(text=None, images=[image1, image2], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            image_features = clip_model.get_image_features(**inputs)
            
            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate the cosine similarity
            clip_score = (image_features[0] @ image_features[1].T).item()
            clip_scores[image_name] = clip_score

            # print(f"Processed: {image_name}, CLIP Score: {clip_score:.4f}")

        except Exception as e:
            print(f"Could not process {image_name}. Error: {e}")


    # --- Compute Final FID Score ---
    final_fid_score = fid_metric.compute()
    print("\n--- Results ---")
    print(f"Final FID Score: {final_fid_score.item():.4f}")

    if clip_scores:
        average_clip_score = sum(clip_scores.values()) / len(clip_scores)
        print(f"Average CLIP Score: {average_clip_score:.4f}")

    # print("\nIndividual CLIP Scores:")
    # for image_name, score in clip_scores.items():
    #     print(f"  {image_name}: {score:.4f}")
        

if __name__ == '__main__':
    # Calculate CLIP scores between text prompts and images:
    csv_file = '/workspace/NegDiff/data/nudity.csv'
    image_folder = '/workspace/NegDiff/results/SD_only-emp_v1-4_unlearndiff-2/all'
    
    calculate_text_image_clip_scores(csv_file, image_folder)
    
    
    # Calculate FID and CLIP scores between two folders of images:
    # Replace these with the actual paths to your image folders
    # folder1 = '/workspace/NegDiff/results/SD_ortho-neg_v1-4_unlearndiff_7/all'
    # folder2 = '/workspace/NegDiff/results/SD_emp-neg_v1-4_unlearndiff/all'

    # calculate_scores(folder1, folder2)
