import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

class YOLOLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets):
        total_loss = 0
        
        for pred in predictions:
            batch_size, _, grid_h, grid_w = pred.shape
            pred = pred.view(batch_size, 3, 5 + self.num_classes, grid_h, grid_w)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            obj_mask = targets[..., 4] == 1
            no_obj_mask = targets[..., 4] == 0
            
            if obj_mask.sum() > 0:
                box_loss = self.mse_loss(
                    pred[obj_mask][..., :4],
                    targets[obj_mask][..., :4]
                )
                
                obj_loss = self.bce_loss(
                    pred[obj_mask][..., 4],
                    targets[obj_mask][..., 4]
                )
                
                cls_loss = self.bce_loss(
                    pred[obj_mask][..., 5:],
                    targets[obj_mask][..., 5:]
                )
                
                total_loss += box_loss + obj_loss + cls_loss
            
            if no_obj_mask.sum() > 0:
                no_obj_loss = self.bce_loss(
                    pred[no_obj_mask][..., 4],
                    targets[no_obj_mask][..., 4]
                )
                total_loss += 0.5 * no_obj_loss
        
        return total_loss

class ModelTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = YOLOLoss(config['model']['num_classes'])
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        self.train_losses = []
        self.val_losses = []
        self.best_loss = float('inf')
        
    def _create_optimizer(self):
        lr = self.config['training']['learning_rate']
        optimizer_type = self.config['training']['optimizer']
        
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        else:
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
    
    def _create_scheduler(self):
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            images = batch['image'].to(self.device)
            targets = self._prepare_targets(batch)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                targets = self._prepare_targets(batch)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader):
        num_epochs = self.config['training']['epochs']
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_model("best_model.pth")
            
            if (epoch + 1) % 10 == 0:
                self.save_model(f"model_epoch_{epoch+1}.pth")
    
    def _prepare_targets(self, batch):
        batch_size = len(batch['image'])
        grid_sizes = [80, 40, 20]
        targets = []
        
        for grid_size in grid_sizes:
            target = torch.zeros(batch_size, 3, grid_size, grid_size, 5 + self.config['model']['num_classes'])
            targets.append(target.to(self.device))
        
        return targets
    
    def save_model(self, filename):
        model_dir = self.config['paths']['model_dir']
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }, os.path.join(model_dir, filename))
    
    def load_model(self, filename):
        model_dir = self.config['paths']['model_dir']
        checkpoint = torch.load(os.path.join(model_dir, filename), map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
    
    def get_training_history(self):
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }