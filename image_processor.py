"""
image_processor.py

This script processes images by applying a transformation pipeline,
resizing while preserving aspect ratio, adding padding, and converting images to RGB.
It is used to prepare images for feature extraction.
"""

import os         # Module to interact with the operating system.
import sys        # Module to access system-specific parameters and functions.
from typing import Optional  # For type hinting (e.g., Optional return types).

import torch      # PyTorch library for tensor operations.
from PIL import Image  # Python Imaging Library for image processing.
from torchvision import transforms  # Transformations for image preprocessing.

class ImageProcessor:
    """
    ImageProcessor handles image processing tasks such as transforming images,
    resizing with padding, and processing batches of images from a folder.
    """

    def __init__(self, raw_image_folder: str = "images"):
        """
        Initialize the ImageProcessor.

        Args:
            raw_image_folder (str): Folder where raw images are stored. Defaults to "images".
        """
        # Ensure that the raw image folder exists; if not, create it.
        os.makedirs("images", exist_ok=True)
        self.image_folder = raw_image_folder

    def transformer(self):
        """
        Creates a transformation pipeline for image processing.

        The pipeline performs the following steps:
          1. Resize the image to 256x256 pixels.
          2. Convert the image to a PyTorch tensor.
          3. Normalize the tensor using ImageNet mean and standard deviation.

        Returns:
            A torchvision.transforms.Compose object that applies the above transformations.
        """
        pipeline_transformer = transforms.Compose([
            # Resize the image to 256x256.
            transforms.Resize((256, 256)),
            # Convert the PIL image to a PyTorch tensor.
            transforms.ToTensor(),
            # Normalize the tensor with mean and std values suitable for pretrained networks.
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return pipeline_transformer

    def process_image(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Load an image from a given path, apply the transformation pipeline, and add a batch dimension.

        Args:
            image_path (str): Path to the image file.

        Returns:
            Optional[torch.Tensor]: Processed image tensor with an extra batch dimension on success;
                                    None if an error occurs.
        """
        try:
            # Open the image file and ensure it is in RGB format.
            image = Image.open(image_path).convert("RGB")
            # Get the transformation pipeline.
            transformer = self.transformer()
            # Apply the transformation to the image.
            img_tensor = transformer(image)
            # Add an extra batch dimension (e.g., from [C, H, W] to [1, C, H, W]).
            return img_tensor.unsqueeze(0)
        except FileNotFoundError as e:
            # Print error message if file is not found.
            print(e)
            print(f"Can't find the image file with path\n {image_path}")
        except Exception as e:
            # Catch any other exceptions and print an error message.
            print(f"Error opening image {image_path}: {e}")
            return

    def get_images_fps(self) -> list:
        """
        Retrieve a list of valid image filenames from the specified image folder.

        Returns:
            list: List of image filenames (with valid extensions) found in the image folder.
        """
        # Define valid image file extensions.
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        # List all files in the image folder and filter by valid extensions.
        image_files = [f for f in os.listdir(self.image_folder) if f.lower().endswith(valid_extensions)]
        # If no valid images are found, print an error and exit the program.
        if image_files == []:
            print(f"No valid images found in '{self.image_folder}'.")
            sys.exit(1)
        return image_files

    def test_image_processing(self, image_file_name: str):
        """
        Tests the image processing pipeline on a single image file and prints the processed image shape.

        Args:
            image_file_name (str): The filename of the image to be processed.
        """
        # Sort the image file name (if multiple characters, this may not be necessary, but ensures consistency).
        image_filename: str = sorted(image_file_name)
        # Construct the full path to the image file.
        image_path: str = os.path.join(self.image_folder, image_filename)
        # Inform which image is being processed.
        print(f"Processing first image: {image_filename}")
        # Process the image using the process_image method.
        processed_image = self.process_image(image_path)
        # If image processing failed, terminate the program.
        if processed_image is None:
            sys.exit(1)
        # Print the shape of the processed image tensor.
        print(f"Processed image shape: {processed_image.shape}")

    def resize_image(self, final_size: int, img: Image.Image) -> Image.Image:
        """
        Resize an image while maintaining its aspect ratio and pad the image to match the target size.

        Args:
            final_size (int): Target size for both width and height (square output).
            img (Image.Image): The original PIL image.

        Returns:
            Image.Image: The resized and padded image.
        """
        # Get the original size (width, height) of the image.
        size = img.size
        # Calculate the scaling factor based on the larger dimension.
        ratio = float(final_size) / max(size)
        # Compute the new size while preserving the aspect ratio.
        new_image_size = tuple([int(x * ratio) for x in size])
        # Resize the image using a high-quality downsampling filter.
        img = img.resize(new_image_size, Image.Resampling.LANCZOS)
        # Create a new black image with the target final size.
        new_img = Image.new("RGB", (final_size, final_size), (0, 0, 0))
        # Calculate position to center the resized image on the new black background.
        paste_x = (final_size - new_image_size[0]) // 2
        paste_y = (final_size - new_image_size[1]) // 2
        # Paste the resized image onto the center of the new image.
        new_img.paste(img, (paste_x, paste_y))
        return new_img

    def process_image_data(
        self,
        output_folder: str = "cleaned_images",
        final_size: int = 256  # Target output dimensions.
    ) -> None:
        """
        Process all images: resize (with padding) and convert each image to RGB, then save the processed images to a new folder.

        Args:
            output_folder (str): Folder where cleaned/processed images will be saved.
            final_size (int): Desired final image dimensions.
        """
        # Create the output folder if it does not exist.
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Get a list of image filenames from the input image folder.
        img_paths = self.get_images_fps()

        # Process each image in the folder.
        for img_path in img_paths:
            try:
                # Build the full path to the current image.
                full_img_path = os.path.join(self.image_folder, img_path)
                # Open the image file.
                with Image.open(full_img_path) as img:
                    # Resize the image with padding.
                    new_im = self.resize_image(final_size, img)
                    # Define the output file path.
                    output_file_path = os.path.join(output_folder, img_path)
                    # Save the processed image.
                    new_im.save(output_file_path)
                    print(f"Saved cleaned image to: {output_file_path}")
            except Exception as e:
                # Inform the user if any error occurs while processing an image.
                print(f"Skipping file '{img_path}' due to error: {e}")