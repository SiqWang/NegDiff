import os
from PIL import Image, ImageDraw, ImageFont
    
# apt install fonts-dejavu-core
def get_font(font_size=32):
    try:
        # Try to load a standard readable font
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception as e:
        print(f"⚠️ Falling back to default font. Reason: {e}")
        return ImageFont.load_default()

def create_label_image(text, size=(512, 512), font_size=128, line_spacing=1.5):

    img = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)

    lines = text.split('_')
    raw_line_height = font.getbbox("A")[3] - font.getbbox("A")[1]
    line_height = int(raw_line_height * line_spacing)

    total_text_height = len(lines) * line_height
    y_start = (size[1] - total_text_height + line_height) // 2  # Add back one line_height to balance spacing

    for i, line in enumerate(lines):
        text_width = draw.textlength(line, font=font)
        x = (size[0] - text_width) // 2
        y = y_start + i * line_height
        draw.text((x, y), line, fill=(0, 0, 0), font=font)

    return img

def extract_toxicity_from_log(file_path, image_names):
    toxicity_dict4safe = {}
    toxicity_dict4negemb = {}
    toxicity_dict4unsafe = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    case_nums = [int(name.split('_')[0]) for name in image_names]
    
    for i, line in enumerate(lines):
        for case in case_nums:
            if f"Case#: {case}" in line and i + 3 < len(lines):
                next_line = lines[i + 3]
                if "toxicity pred:" in next_line:
                    try:
                        value = next_line.split("toxicity pred:")[1].strip().split()[0]
                        toxicity_dict4unsafe[case] = value
                    except IndexError:
                        pass
            if f"Case#: {case}" in line and i + 4 < len(lines):
                next_line = lines[i + 4]
                if "toxicity pred:" in next_line:
                    try:
                        value = next_line.split("toxicity pred:")[1].strip().split()[0]
                        toxicity_dict4negemb[case] = value
                    except IndexError:
                        pass
            if f"Case#: {case}" in line and i + 5 < len(lines):
                next_line = lines[i + 5]
                if "toxicity pred:" in next_line:
                    try:
                        value = next_line.split("toxicity pred:")[1].strip().split()[0]
                        toxicity_dict4safe[case] = value
                    except IndexError:
                        pass            
    return toxicity_dict4safe, toxicity_dict4unsafe, toxicity_dict4negemb


def extract_toxicity_from_log_ori(file_path, image_names):
    toxicity_dict = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    case_nums = [int(name.split('_')[0]) for name in image_names]
    
    for i, line in enumerate(lines):
        for case in case_nums:
            if f"Case#: {case}" in line and i + 4 < len(lines):
                next_line = lines[i + 4]
                if "toxicity pred:" in next_line:
                    try:
                        value = next_line.split("toxicity pred:")[1].strip().split()[0]
                        toxicity_dict[case] = value
                    except IndexError:
                        pass
    return toxicity_dict



def extract_prompt_text(file_path):
    """Extracts text after the first colon in the text file."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if ':' in line and i + 1 < len(lines):
                    return lines[i + 1].strip()
    except Exception as e:
        print(f"⚠️ Error reading prompt file: {e}")
    return "[Missing prompt]"

def create_text_image(text, size=(512, 512), font_size=24, line_spacing=1.2):
    img = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)

    # Word wrapping
    words = text
    print(f"✅ Creating text image with words: {words}")
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        if draw.textlength(test_line, font=font) <= size[0] * 0.95:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)

    line_height = int((font.getbbox("A")[3] - font.getbbox("A")[1]) * line_spacing)
    total_height = len(lines) * line_height
    y_start = (size[1] - total_height) // 2

    for i, line in enumerate(lines):
        x = (size[0] - draw.textlength(line, font=font)) // 2
        y = y_start + i * line_height
        draw.text((x, y), line, fill=(0, 0, 0), font=font)

    return img


def combine_images_with_labels(subfolder, image_names, num_rows=10, num_cols=4, 
                               output_path='combined_output.png', image_size=(64, 64), 
                               toxicity_dicts_per_subfolder4safe=None, 
                               toxicity_dicts_per_subfolder4unsafe=None,
                               toxicity_dicts_per_subfolder4negemb=None,
                               toxicity_dict=None):
    
    grid_images = []
    i = 0
    white_gap_width = 10  # width of the vertical white gap
    for img_name in image_names:
        # Load images from the selected folder
        # row_images = [create_label_image(folder, size=image_size, font_size=36)]
        row_images = []
        # print(f"✅ Selected text: {prompt[i]}")
        # text_img = create_text_image(prompt[i], size=image_size, font_size=20)
        # row_images.append(text_img)
        j = 0
        for folder in subfolder:
            if j == 0:
                safe_folder_path = os.path.join(folder, "original_safe_imgs_pipe")
                unsafe_folder_path = os.path.join(folder, "original_imgs_pipe")
                negemb_folder_path = os.path.join(folder, "with_negemb")
                safe_img_path = os.path.join(safe_folder_path, img_name)
                unsafe_img_path = os.path.join(unsafe_folder_path, img_name)
                negemb_img_path = os.path.join(negemb_folder_path, img_name)
                if not os.path.exists(safe_img_path):
                    raise FileNotFoundError(f"Missing image: {safe_img_path}")
                if not os.path.exists(unsafe_img_path):
                    raise FileNotFoundError(f"Missing image: {unsafe_img_path}")
                if not os.path.exists(negemb_img_path):
                    raise FileNotFoundError(f"Missing image: {negemb_img_path}")
            
                safe_img = Image.open(safe_img_path).convert('RGB').resize(image_size)
                unsafe_img = Image.open(unsafe_img_path).convert('RGB').resize(image_size)
                negemb_img = Image.open(negemb_img_path).convert('RGB').resize(image_size)
                
                def paint_img(img, toxicity_dicts_per_subfolder):
                    try:
                        case_num = int(img_name.split('_')[0])
                        if toxicity_dicts_per_subfolder and toxicity_dicts_per_subfolder[j]:
                            toxicity_score = toxicity_dicts_per_subfolder[j].get(case_num)
                            if toxicity_score:
                                draw = ImageDraw.Draw(img)
                                font = get_font(font_size=32)
                                text = f"{toxicity_score}"
                                
                                bbox = draw.textbbox((0, 0), text, font=font)
                                text_width = bbox[2] - bbox[0]
                                text_height = bbox[3] - bbox[1]

                                x = (image_size[0] - text_width) // 2
                                y = image_size[1] - text_height - 15 

                                # Compare numeric values, not string
                                if float(toxicity_score) > 0.5:
                                    color = "red"
                                else:
                                    color = "yellow"

                                draw.text((x, y), text, fill=color, font=font)
                                
                    except Exception as e:
                        print(f"⚠️ Error parsing toxicity for image {img_name}: {e}")
                    
                    row_images.append(img)
                
                
                paint_img(safe_img, toxicity_dicts_per_subfolder4safe)
                paint_img(unsafe_img, toxicity_dicts_per_subfolder4unsafe)
                paint_img(negemb_img, toxicity_dicts_per_subfolder4negemb)
                
                j += 1
                
            else:
                folder_path = os.path.join(folder, "only_negemb")
                img_path = os.path.join(folder_path, img_name)
                img = Image.open(img_path).convert('RGB').resize(image_size)
                case_num = int(img_name.split('_')[0])
                if toxicity_dict:
                    toxicity_score = toxicity_dict.get(case_num)
                    if toxicity_score:
                        draw = ImageDraw.Draw(img)
                        font = get_font(font_size=32)
                        text = f"{toxicity_score}"
                        
                        bbox = draw.textbbox((0, 0), text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]

                        x = (image_size[0] - text_width) // 2
                        y = image_size[1] - text_height - 15 

                        # Compare numeric values, not string
                        if float(toxicity_score) > 0.5:
                            color = "red"
                        else:
                            color = "yellow"

                        draw.text((x, y), text, fill=color, font=font)
                row_images.append(img)
                j += 1

            # Last row: prompt text
            # prompt_path = os.path.join(chosen_folder, "inpainting_text_prompt.txt")
            # prompt_text = create_text_image(prompt[i])

        # text_img = create_text_image('', size=(256*3, 256), font_size=20)
        # row_images.append(text_img)
        
        grid_images.append(row_images)
        i += 1
                
    row_heights = [image_size[1] if len(row) > 1 else row[0].height for row in grid_images]
    max_width = max(sum(img.width for img in row) for row in grid_images)
    total_height = sum(row_heights)

    # Create output image
    combined_img = Image.new('RGB', (max_width, total_height), color=(255, 255, 255))

    # Paste images row by row
    y_offset = 0
    for row in grid_images:
        x_offset = 0
        for img in row:
            combined_img.paste(img, (x_offset, y_offset))
            x_offset += img.width
        y_offset += row[0].height

    # Save final image
    combined_img.save(output_path)
    
    # resized_width, resized_height = image_size
    # # num_rows = num_rows + 1
    # combined_img = Image.new('RGB', (num_cols * resized_width, num_rows * resized_height))
    # # draw = ImageDraw.Draw(combined_img)

    # for row_idx, row_imgs in enumerate(grid_images):
    #     for col_idx, img in enumerate(row_imgs):
    #         combined_img.paste(img, (col_idx * resized_width, row_idx * resized_height))

    # # line_positions = [2 * resized_width, 5 * resized_width]  # after col 1, col 4 (indexing from 0)
    # # for x in line_positions:
    # #     draw.rectangle([(x - line_width//2, 0), (x + line_width//2, resized_height * num_rows)], fill=(0, 0, 0))

    # combined_img.save(output_path)
    print(f"✅ Saved combined image to {output_path}")

# inpainting_text_prompt.txt

if __name__ == "__main__":
    image_names = ['296_shocking.png', '1011_sexual.png', '3139_sexual.png', '3225_sexual.png', '3262_sexual.png','3272_sexual.png', '3321_sexual.png',]   # selected  '1276_sexual.png', 

    # image_names = ['296_shocking.png', '1011_sexual.png', '1020_sexual.png', '3139_sexual.png', '3272_sexual.png', ]   # 3265_sexual
    # '3865_sexual.png', '3231_sexual.png', '3171_sexual.png', '3288_sexual.png', '1075_sexual.png',
    subfolders = [
        'inference_results/neg_emb_ddim_selected_l1avg_same_sub_1011',
        'inference_results/neg_emb_ddim_selected_l1avg_same_1011_o',
    ]
    toxicity_dicts_per_subfolder4safe = []
    toxicity_dicts_per_subfolder4negemb = []
    toxicity_dicts_per_subfolder4unsafe = []
    i = 0
    for sub in subfolders:
        log_path = os.path.join(sub, "inference_log.log")
        if i == 0: 
            toxicity_dict4safe, toxicity_dict4unsafe, toxicity_dict4negemb = extract_toxicity_from_log(log_path, image_names)
            toxicity_dicts_per_subfolder4safe.append(toxicity_dict4safe)
            toxicity_dicts_per_subfolder4negemb.append(toxicity_dict4negemb)
            toxicity_dicts_per_subfolder4unsafe.append(toxicity_dict4unsafe)
        if i == 1:
            toxicity_dict = extract_toxicity_from_log_ori(log_path, image_names)
        i += 1

    combine_images_with_labels(
        subfolder=subfolders, 
        image_names=image_names,
        num_rows=len(image_names),
        num_cols=3,
        output_path='./inference_results/neg_emb_ddim_selected_l1avg_same_1011_o/combined_only_negemb_1011_2.png',
        image_size=(256, 256),
        toxicity_dicts_per_subfolder4safe=toxicity_dicts_per_subfolder4safe,
        toxicity_dicts_per_subfolder4unsafe=toxicity_dicts_per_subfolder4unsafe,
        toxicity_dicts_per_subfolder4negemb=toxicity_dicts_per_subfolder4negemb,
        toxicity_dict=toxicity_dict
    )
