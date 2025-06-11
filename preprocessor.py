import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

class ImagePreprocessor:
    def __init__(self, config):
        self.input_size = tuple(config['model']['input_size'])
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def resize_image(self, image, target_size=None):
        if target_size is None:
            target_size = self.input_size
        return cv2.resize(image, target_size)
    
    def normalize_image(self, image):
        image = image.astype(np.float32) / 255.0
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]
        return image
    
    def enhance_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    def reduce_noise(self, image):
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def preprocess_single(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = self.resize_image(image)
        image = self.enhance_contrast(image)
        image = self.reduce_noise(image)
        image = self.normalize_image(image)
        
        return image
    
    def preprocess_batch(self, images):
        processed = []
        for img in images:
            processed.append(self.preprocess_single(img))
        return np.array(processed)
    
    def create_transform(self, is_training=True):
        transform_list = [
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ]
        
        if is_training:
            transform_list.insert(-2, transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ))
        
        return transforms.Compose(transform_list)
    
    def denormalize(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
    
    def tensor_to_image(self, tensor):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        tensor = self.denormalize(tensor.clone())
        tensor = torch.clamp(tensor, 0, 1)
        
        image = tensor.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        
        return image