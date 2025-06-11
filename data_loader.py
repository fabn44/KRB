import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from PIL import Image

class MaritimeDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ann_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        ann_path = os.path.join(self.annotations_dir, ann_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) == 5:
                        labels.append(int(data[0]))
                        boxes.append([float(x) for x in data[1:]])
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_name': img_name
        }

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.data_dir = config['paths']['data_dir']
        
    def load_dataset(self, subset='train'):
        images_dir = os.path.join(self.data_dir, subset, 'images')
        annotations_dir = os.path.join(self.data_dir, subset, 'labels')
        return MaritimeDataset(images_dir, annotations_dir)
    
    def split_dataset(self, source_dir):
        images_dir = os.path.join(source_dir, 'images')
        labels_dir = os.path.join(source_dir, 'labels')
        
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        np.random.shuffle(image_files)
        
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['val_ratio']
        
        train_split = int(len(image_files) * train_ratio)
        val_split = int(len(image_files) * (train_ratio + val_ratio))
        
        splits = {
            'train': image_files[:train_split],
            'val': image_files[train_split:val_split],
            'test': image_files[val_split:]
        }
        
        for split_name, files in splits.items():
            self._create_split_dirs(split_name, files, images_dir, labels_dir)
    
    def _create_split_dirs(self, split_name, files, source_images, source_labels):
        split_dir = os.path.join(self.data_dir, split_name)
        os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)
        
        for img_file in files:
            src_img = os.path.join(source_images, img_file)
            dst_img = os.path.join(split_dir, 'images', img_file)
            
            label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
            src_label = os.path.join(source_labels, label_file)
            dst_label = os.path.join(split_dir, 'labels', label_file)
            
            if os.path.exists(src_img):
                os.system(f'cp "{src_img}" "{dst_img}"')
            if os.path.exists(src_label):
                os.system(f'cp "{src_label}" "{dst_label}"')
    
    def get_class_distribution(self, subset='train'):
        dataset = self.load_dataset(subset)
        class_counts = {i: 0 for i in range(self.config['model']['num_classes'])}
        
        for i in range(len(dataset)):
            sample = dataset[i]
            for label in sample['labels']:
                class_counts[label.item()] += 1
                
        return class_counts