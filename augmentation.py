import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

class DataAugmentation:
    def __init__(self, config):
        self.config = config
        self.input_size = tuple(config['model']['input_size'])
        
    def get_training_transform(self):
        return A.Compose([
            A.Resize(height=self.input_size[1], width=self.input_size[0]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2, 
                rotate_limit=15, 
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20, 
                sat_shift_limit=30, 
                val_shift_limit=20, 
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    
    def get_validation_transform(self):
        return A.Compose([
            A.Resize(height=self.input_size[1], width=self.input_size[0]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    
    def augment_single_image(self, image, bboxes, labels):
        transform = self.get_training_transform()
        
        augmented = transform(
            image=image,
            bboxes=bboxes,
            class_labels=labels
        )
        
        return {
            'image': augmented['image'],
            'bboxes': augmented['bboxes'],
            'labels': augmented['class_labels']
        }
    
    def create_augmented_dataset(self, original_images, original_bboxes, original_labels, multiplier=3):
        augmented_images = []
        augmented_bboxes = []
        augmented_labels = []
        
        for i in range(len(original_images)):
            for _ in range(multiplier):
                aug_data = self.augment_single_image(
                    original_images[i],
                    original_bboxes[i],
                    original_labels[i]
                )
                
                augmented_images.append(aug_data['image'])
                augmented_bboxes.append(aug_data['bboxes'])
                augmented_labels.append(aug_data['labels'])
        
        return augmented_images, augmented_bboxes, augmented_labels
    
    def apply_cutmix(self, image1, image2, bboxes1, bboxes2, labels1, labels2, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        
        h, w = image1.shape[:2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        mixed_image = image1.copy()
        mixed_image[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]
        
        mixed_bboxes = list(bboxes1) + list(bboxes2)
        mixed_labels = list(labels1) + list(labels2)
        
        return mixed_image, mixed_bboxes, mixed_labels
    
    def apply_mosaic(self, images, bboxes_list, labels_list):
        h, w = self.input_size[1], self.input_size[0]
        
        mosaic_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        positions = [
            (0, 0, h//2, w//2),
            (0, w//2, h//2, w),
            (h//2, 0, h, w//2),
            (h//2, w//2, h, w)
        ]
        
        all_bboxes = []
        all_labels = []
        
        for i, (y1, x1, y2, x2) in enumerate(positions):
            if i < len(images):
                img = cv2.resize(images[i], (x2-x1, y2-y1))
                mosaic_image[y1:y2, x1:x2] = img
                
                scale_x = (x2 - x1) / w
                scale_y = (y2 - y1) / h
                offset_x = x1 / w
                offset_y = y1 / h
                
                for bbox, label in zip(bboxes_list[i], labels_list[i]):
                    new_bbox = [
                        bbox[0] * scale_x + offset_x,
                        bbox[1] * scale_y + offset_y,
                        bbox[2] * scale_x,
                        bbox[3] * scale_y
                    ]
                    all_bboxes.append(new_bbox)
                    all_labels.append(label)
        
        return mosaic_image, all_bboxes, all_labels