"""
image_processor.py

This script processes the first valid image found in the 'cleaned_images' folder by applying the transformation pipeline
and adding a batch dimension. This prepares the image for the feature extraction model.

Key differences from my initial code:
  - Inline comments are added for clarity.
"""

import os
import sys
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms


class ImageProcessor:


    def __init__(self, raw_image_folder: str = "images"):
        os.makedirs("images", exist_ok=True)
        self.image_folder = raw_image_folder

    def transformer(self):
        """
        The function `transform_pipeline` creates a transformation pipeline for image processing in
        PyTorch.
        
        Returns:
          The `transform_pipeline` method returns a composition of transformations that includes
        resizing the image to (256, 256) pixels, converting it to a PyTorch tensor, and normalizing the
        image using the specified mean and standard deviation values.
        """
        pipeline_transformer = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        return pipeline_transformer


    def process_image(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Load an image, apply the transformation pipeline, and add a batch dimension.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            transformer = self.transformer()
            img_tensor = transformer(image)
            return img_tensor.unsqueeze(0)
        except FileNotFoundError as e:
            print(e)
            print(f"Can't find the image file with path\n {image_path}")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return
        
    def get_images_fps(self) -> list: 
        """
        The function `get_images_fps` returns a list of image files with valid extensions from a
        specified folder.
        
        Returns:
          image_files (list): A list of filepaths to the images 
        """
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(valid_extensions)]
        if image_files == []:
            print(f"No valid images found in '{self.image_folder}'.")
            sys.exit(1)
        return image_files
    
    def test_image_processing(self, image_file_name: str):
        """
        The function `test_image_processing` processes an image file located in a specified folder and
        prints information about the processed image.
        
        Args:
            image_file_name (str): The `image_file_name` parameter is a string that represents the name of
            an image file.
        """
        image_filename: str = sorted(image_file_name)
        image_path: str = os.path.join(self.image_folder, image_filename)
        print(f"Processing first image: {image_filename}")
        processed_image = self.process_image(image_path)
        if processed_image is None:
            sys.exit(1)
        print(f"Processed image shape: {processed_image.shape}")

    def resize_image(self, final_size: int, img: Image.Image) -> Image.Image:
        """
        Resize an image while maintaining aspect ratio and add padding to match the target size.
        
        Args:
            final_size (int): Target size for both width and height.
            im (Image.Image): The original image.
            
        Returns:
            Image.Image: The resized and padded image.
        """
        size = img.size  # Original (width, height)
        ratio = float(final_size) / max(size)  # Scale factor based on the largest dimension.
        new_image_size = tuple([int(x * ratio) for x in size])  # New size maintaining aspect ratio.

        img = img.resize(new_image_size, Image.LANCZOS)  # Resize with high-quality filter.
        new_img = Image.new("RGB", (final_size, final_size), (0, 0, 0))  # Create a black background.
        # Center the resized image on the new background.
        new_img.paste(img, ((final_size - new_image_size[0]) // 2, (final_size - new_image_size[1]) // 2))
        return new_img

    def process_image_data(
        self,
        output_folder: str = "cleaned_images",
        final_size: int = 256  # Adjust this value as needed.
    ) -> None:
        """
        Process images by resizing (with padding) and converting to RGB, then save them to a new folder.
        
        Args:
            input_folder (str): Directory containing the original images.
            output_folder (str): Directory to save cleaned images.
            final_size (int): Target size for the images.
        """
        # Ensure the output folder exists.
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        img_paths = self.get_images_fps()

        # Process each file in the input folder.
        for img_path in img_paths:
            try:
                with Image.open(self.image_folder + "/" + img_path) as img:
                    # Resize and pad the image.
                    new_im = self.resize_image(final_size, img)

                    # Save the processed image in the output folder.
                    output_file_path = os.path.join(output_folder, img_path)
                    new_im.save(output_file_path)
                    print(f"Saved cleaned image to: {output_file_path}")
            except Exception as e:
                print(f"Skipping file '{img_path}' due to error: {e}")
