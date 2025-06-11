from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QProgressBar, QTextEdit, QGroupBox, 
                             QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import torch
from torch.utils.data import DataLoader

from ..models.yolo_model import create_yolo_model
from ..models.trainer import ModelTrainer
from ..data.data_loader import MaritimeDataset
from ..data.augmentation import DataAugmentation

class TrainingThread(QThread):
    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    epoch_finished = pyqtSignal(int, float, float)
    training_finished = pyqtSignal(bool)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def run(self):
        try:
            self.log_updated.emit("Initializing model...")
            model = create_yolo_model(self.config['model']['num_classes'])
            trainer = ModelTrainer(model, self.config)
            
            self.log_updated.emit("Loading datasets...")
            augmentation = DataAugmentation(self.config)
            
            train_dataset = MaritimeDataset(
                f"{self.config['paths']['data_dir']}/train/images",
                f"{self.config['paths']['data_dir']}/train/labels",
                transform=augmentation.get_training_transform()
            )
            
            val_dataset = MaritimeDataset(
                f"{self.config['paths']['data_dir']}/val/images",
                f"{self.config['paths']['data_dir']}/val/labels",
                transform=augmentation.get_validation_transform()
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=2
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=2
            )
            
            self.log_updated.emit(f"Training set: {len(train_dataset)} images")
            self.log_updated.emit(f"Validation set: {len(val_dataset)} images")
            self.log_updated.emit("Starting training...")
            
            num_epochs = self.config['training']['epochs']
            
            for epoch in range(num_epochs):
                self.log_updated.emit(f"Epoch {epoch+1}/{num_epochs}")
                
                train_loss = trainer.train_epoch(train_loader)
                val_loss = trainer.validate_epoch(val_loader)
                
                trainer.train_losses.append(train_loss)
                trainer.val_losses.append(val_loss)
                trainer.scheduler.step()
                
                progress = int(((epoch + 1) / num_epochs) * 100)
                self.progress_updated.emit(progress)
                
                self.epoch_finished.emit(epoch + 1, train_loss, val_loss)
                
                if val_loss < trainer.best_loss:
                    trainer.best_loss = val_loss
                    trainer.save_model("best_model.pth")
                    self.log_updated.emit("Saved best model")
                
                if (epoch + 1) % 10 == 0:
                    trainer.save_model(f"model_epoch_{epoch+1}.pth")
            
            trainer.save_model("final_model.pth")
            self.log_updated.emit("Training completed successfully!")
            self.training_finished.emit(True)
            
        except Exception as e:
            self.log_updated.emit(f"Training failed: {str(e)}")
            self.training_finished.emit(False)

class TrainDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        
        self.setWindowTitle("Model Training")
        self.setGeometry(200, 200, 700, 600)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        params_group = QGroupBox("Training Parameters")
        params_layout = QVBoxLayout()
        
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(self.config['training']['epochs'])
        epochs_layout.addWidget(self.epochs_spin)
        params_layout.addLayout(epochs_layout)
        
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(self.config['training']['batch_size'])
        batch_layout.addWidget(self.batch_spin)
        params_layout.addLayout(batch_layout)
        
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setValue(self.config['training']['learning_rate'])
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.0001)
        lr_layout.addWidget(self.lr_spin)
        params_layout.addLayout(lr_layout)
        
        optimizer_layout = QHBoxLayout()
        optimizer_layout.addWidget(QLabel("Optimizer:"))
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(['adam', 'sgd'])
        self.optimizer_combo.setCurrentText(self.config['training']['optimizer'])
        optimizer_layout.addWidget(self.optimizer_combo)
        params_layout.addLayout(optimizer_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        device_group = QGroupBox("Device Information")
        device_layout = QVBoxLayout()
        
        device_available = "CUDA available" if torch.cuda.is_available() else "CPU only"
        device_label = QLabel(f"Device: {device_available}")
        device_layout.addWidget(device_label)
        
        if torch.cuda.is_available():
            gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
            device_layout.addWidget(QLabel(gpu_info))
        
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)
        
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.epoch_label = QLabel("Epoch: 0/0")
        progress_layout.addWidget(self.epoch_label)
        
        self.loss_label = QLabel("Train Loss: 0.0000 | Val Loss: 0.0000")
        progress_layout.addWidget(self.loss_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        button_layout.addWidget(self.stop_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def start_training(self):
        self.config['training']['epochs'] = self.epochs_spin.value()
        self.config['training']['batch_size'] = self.batch_spin.value()
        self.config['training']['learning_rate'] = self.lr_spin.value()
        self.config['training']['optimizer'] = self.optimizer_combo.currentText()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        self.training_thread = TrainingThread(self.config)
        
        self.training_thread.progress_updated.connect(self.progress_bar.setValue)
        self.training_thread.log_updated.connect(self.log_text.append)
        self.training_thread.epoch_finished.connect(self.update_epoch_info)
        self.training_thread.training_finished.connect(self.training_finished)
        
        self.training_thread.start()
    
    def stop_training(self):
        if hasattr(self, 'training_thread') and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.training_finished(False)
    
    def update_epoch_info(self, epoch, train_loss, val_loss):
        total_epochs = self.epochs_spin.value()
        self.epoch_label.setText(f"Epoch: {epoch}/{total_epochs}")
        self.loss_label.setText(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    def training_finished(self, success):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if success:
            self.log_text.append("Training completed successfully!")
        else:
            self.log_text.append("Training stopped or failed!")