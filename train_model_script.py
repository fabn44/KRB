#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import argparse

from src.utils.config_manager import ConfigManager
from src.models.yolo_model import create_yolo_model
from src.models.trainer import ModelTrainer
from src.data.data_loader import MaritimeDataset
from src.data.augmentation import DataAugmentation

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for maritime object detection')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    print("Maritime Object Detection - Training Script")
    print(f"Configuration: {args.config}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print()
    
    model = create_yolo_model(config['model']['num_classes'])
    trainer = ModelTrainer(model, config)
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_model(args.resume)
    
    augmentation = DataAugmentation(config)
    
    train_dataset = MaritimeDataset(
        f"{config['paths']['data_dir']}/train/images",
        f"{config['paths']['data_dir']}/train/labels",
        transform=augmentation.get_training_transform()
    )
    
    val_dataset = MaritimeDataset(
        f"{config['paths']['data_dir']}/val/images",
        f"{config['paths']['data_dir']}/val/labels",
        transform=augmentation.get_validation_transform()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    print(f"Training set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")
    print("Starting training...")
    print()
    
    trainer.train(train_loader, val_loader)
    
    print("Training completed!")
    print(f"Best validation loss: {trainer.best_loss:.4f}")
    
    history = trainer.get_training_history()
    print(f"Final training loss: {history['train_losses'][-1]:.4f}")
    print(f"Final validation loss: {history['val_losses'][-1]:.4f}")

if __name__ == '__main__':
    main()