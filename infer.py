import torch
from PIL import Image
import numpy as np
from distillanydepth.modeling.archs.dam.dam import DepthAnything
from distillanydepth.utils.image_util import chw2hwc, colorize_depth_maps
from distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import os
from tqdm import tqdm

# Image processing function
def process_image(image, model, device, result_gray=False):
    try:
        if model is None:
            return None
        
        # 确保图像是 RGB 格式
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Preprocess the image
        image_np = np.array(image)[..., ::-1] / 255

        transform = Compose([
            Resize(512, 512, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet()
        ])

        image_tensor = transform({'image': image_np})['image']
        image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(device)

        with torch.no_grad():  # Disable autograd since we don't need gradients on CPU
            pred_disp, _ = model(image_tensor)
        torch.cuda.empty_cache()

        # Ensure the depth map is in the correct shape before colorization
        pred_disp_np = pred_disp.cpu().detach().numpy()[0, 0, :, :]  # Remove extra singleton dimensions

        # Normalize depth map
        pred_disp = (pred_disp_np - pred_disp_np.min()) / (pred_disp_np.max() - pred_disp_np.min())

        depth_hwc = None
        if not result_gray:
            # Colorize depth map
            cmap = "Spectral_r"
            depth_colored = colorize_depth_maps(pred_disp[None, ..., None], 0, 1, cmap=cmap).squeeze()  # Ensure correct dimension
            # Convert to uint8 for image display
            depth_colored = (depth_colored * 255).astype(np.uint8)
            # Convert to HWC format (height, width, channels)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_hwc = depth_colored_hwc
        else:
            # Gray depth map
            depth_gray = (pred_disp * 255).astype(np.uint8)
            depth_gray_hwc = np.stack([depth_gray] * 3, axis=-1)  # Convert to 3-channel grayscale
            depth_hwc = depth_gray_hwc

        # Resize to match the original image dimensions (height, width)
        h, w = image_np.shape[:2]
        depth_hwc = cv2.resize(depth_hwc, (w, h), cv2.INTER_LINEAR)

        # Convert to a PIL image
        depth_image = Image.fromarray(depth_hwc)
        return depth_image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Function to find all image files in a directory and its subdirectories
def find_images(root_dir):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    return image_files

# Main function to process all images
def main(input_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_kwargs = dict(
        vitb=dict(
            encoder='vitb',
            features=128,
            out_channels=[96, 192, 384, 768],
        ),
        vitl=dict(
            encoder="vitl",
            features=256,
            out_channels=[256, 512, 1024, 1024],
            use_bn=False,
            use_clstoken=False,
            max_depth=150.0,
            mode='disparity',
            pretrain_type='dinov2',
            del_mask_token=False
        )
    )
    # Load model
    try:
        model = DepthAnything(**model_kwargs['vitl']).to(device)
        # if use hf_hub_download, you can use the following code
        checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"large/model.safetensors", repo_type="model")

        # if use local path, you can use the following code
        # checkpoint_path = "path/to/your/model.safetensors"

        model_weights = load_file(checkpoint_path)
        model.load_state_dict(model_weights)
        model = model.to(device)
    except Exception as e:
        print(f"Model loading failed: {e}")
        return

    # Find all image files in the input directory
    image_files = find_images(input_dir)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image with a progress bar
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Open the image
            image = Image.open(image_file)

            # Process the image
            depth_image = process_image(image, model, device, result_gray=True)

            if depth_image is not None:
                # Get the relative path of the image file
                relative_path = os.path.relpath(image_file, input_dir)
                output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                # Save the depth image
                output_file = os.path.join(output_subdir, os.path.basename(image_file))
                depth_image.save(output_file)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

if __name__ == "__main__":
    input_directory = r"E:\Downloads\fal_test\image_txt"  # Replace with the actual input directory path
    output_directory = r"E:\Downloads\fal_test-depth"  # Replace with the actual output directory path
    main(input_directory, output_directory)