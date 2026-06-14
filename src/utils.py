"""
Utility Functions Module

Helper functions for the face recognition system including:
- Image processing
- Model downloading
- Dataset management
- Database utilities
"""

import cv2
import numpy as np
import logging
import os
import urllib.request
import tarfile
from pathlib import Path
from typing import List, Tuple
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageUtils:
    """Image processing utilities"""
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Load image from file
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            np.ndarray: Loaded image in BGR format
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        return image
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> bool:
        """
        Save image to file
        
        Args:
            image (np.ndarray): Image to save
            output_path (str): Output file path
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image)
            logger.info(f"Image saved: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save image: {str(e)}")
            return False
    
    @staticmethod
    def resize_image(image: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image (np.ndarray): Input image
            width (int): Target width (maintains aspect ratio if height not specified)
            height (int): Target height
        
        Returns:
            np.ndarray: Resized image
        """
        h, w = image.shape[:2]
        
        if width is None and height is None:
            return image
        
        if width is None:
            ratio = height / h
            width = int(w * ratio)
        elif height is None:
            ratio = width / w
            height = int(h * ratio)
        
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return resized
    
    @staticmethod
    def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Crop image region
        
        Args:
            image (np.ndarray): Input image
            x, y (int): Top-left coordinates
            width, height (int): Region dimensions
        
        Returns:
            np.ndarray: Cropped image
        """
        return image[y:y+height, x:x+width]
    
    @staticmethod
    def convert_color(image: np.ndarray, color_code: int) -> np.ndarray:
        """
        Convert image color space
        
        Args:
            image (np.ndarray): Input image
            color_code (int): OpenCV color conversion code
        
        Returns:
            np.ndarray: Converted image
        """
        return cv2.cvtColor(image, color_code)
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize image to 0-1 range
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            np.ndarray: Normalized image
        """
        return image.astype(np.float32) / 255.0


class ModelDownloader:
    """Model downloading utilities"""
    
    MODEL_URLS = {
        'lfw_dataset': 'http://vis-www.cs.umass.edu/lfw/lfw.tgz',
        'age_model': 'https://github.com/yu4u/age-gender-estimation/raw/master/age_net.caffemodel',
        'gender_model': 'https://github.com/yu4u/age-gender-estimation/raw/master/gender_net.caffemodel',
    }
    
    @staticmethod
    def download_file(url: str, destination: str) -> bool:
        """
        Download file from URL
        
        Args:
            url (str): URL to download from
            destination (str): Local destination path
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            logger.info(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, destination)
            logger.info(f"Downloaded to {destination}")
            return True
        
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            return False
    
    @staticmethod
    def download_lfw_dataset(destination: str = "data/lfw_dataset") -> bool:
        """
        Download LFW (Labeled Faces in the Wild) dataset
        
        Args:
            destination (str): Destination directory
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            url = ModelDownloader.MODEL_URLS['lfw_dataset']
            tar_file = os.path.join(destination, 'lfw.tgz')
            
            os.makedirs(destination, exist_ok=True)
            
            # Download
            logger.info("Downloading LFW dataset...")
            urllib.request.urlretrieve(url, tar_file)
            
            # Extract
            logger.info("Extracting dataset...")
            with tarfile.open(tar_file, 'r:gz') as tar:
                tar.extractall(destination)
            
            os.remove(tar_file)
            logger.info(f"LFW dataset ready at {destination}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to download LFW dataset: {str(e)}")
            return False
    
    @staticmethod
    def download_pretrained_models(destination: str = "models") -> bool:
        """
        Download pre-trained models
        
        Args:
            destination (str): Destination directory
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            os.makedirs(destination, exist_ok=True)
            
            models = ['age_model', 'gender_model']
            
            for model_name in models:
                if model_name in ModelDownloader.MODEL_URLS:
                    url = ModelDownloader.MODEL_URLS[model_name]
                    file_name = url.split('/')[-1]
                    destination_path = os.path.join(destination, file_name)
                    
                    if not os.path.exists(destination_path):
                        ModelDownloader.download_file(url, destination_path)
            
            logger.info("Pre-trained models downloaded")
            return True
        
        except Exception as e:
            logger.error(f"Failed to download models: {str(e)}")
            return False


class DatabaseUtils:
    """Database utilities for storing face data"""
    
    @staticmethod
    def load_json(file_path: str) -> dict:
        """Load JSON file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON: {str(e)}")
            return {}
    
    @staticmethod
    def save_json(data: dict, file_path: str) -> bool:
        """Save data as JSON"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Data saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON: {str(e)}")
            return False


class FileUtils:
    """File system utilities"""
    
    @staticmethod
    def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
        """
        Get all image files from directory
        
        Args:
            directory (str): Directory path
            extensions (List[str]): File extensions to search for
        
        Returns:
            List[str]: List of image file paths
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        image_files = []
        
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return image_files
        
        for file in os.listdir(directory):
            if os.path.splitext(file)[1].lower() in extensions:
                image_files.append(os.path.join(directory, file))
        
        return image_files
    
    @staticmethod
    def create_directory(directory: str) -> bool:
        """Create directory if it doesn't exist"""
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory: {str(e)}")
            return False
    
    @staticmethod
    def delete_file(file_path: str) -> bool:
        """Delete file safely"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file: {str(e)}")
            return False


def download_models():
    """Download all required models"""
    logger.info("Starting model download...")
    ModelDownloader.download_pretrained_models()
    logger.info("All models downloaded successfully")


def download_lfw():
    """Download LFW dataset"""
    logger.info("Starting LFW dataset download...")
    ModelDownloader.download_lfw_dataset()
    logger.info("LFW dataset downloaded successfully")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Utility functions")
    parser.add_argument('--download-models', action='store_true', help='Download pre-trained models')
    parser.add_argument('--download-lfw', action='store_true', help='Download LFW dataset')
    
    args = parser.parse_args()
    
    if args.download_models:
        download_models()
    
    if args.download_lfw:
        download_lfw()
