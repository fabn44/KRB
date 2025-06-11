from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QProgressBar, QTextEdit, 
                             QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os

from ..data.data_loader import DataLoader
from ..data.preprocessor import ImagePreprocessor
from ..data.augmentation import DataAugmentation
from ..utils.file_utils import FileManager

class DataProcessingThread(QThread):
    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)
    
    def __init__(self, config, source_dir, operations):
        super().__init__()
        self.config = config
        self.source_dir = source_dir
        self.operations = operations
        
    def run(self):
        try:
            file_manager = FileManager(self.config)
            data_loader = DataLoader(self.config)
            preprocessor = ImagePreprocessor(self.config)
            augmentator = DataAugmentation(self.config)
            
            if 'create_structure' in self.operations:
                self.log_updated.emit("Creating project structure...")
                file_manager.create_project_structure()
                self.progress_updated.emit(20)
            
            if 'split_dataset' in self.operations and self.source_dir:
                self.log_updated.emit("Splitting dataset...")
                data_loader.split_dataset(self.source_dir)
                self.progress_updated.emit(50)
            
            if 'preprocess' in self.operations:
                self.log_updated.emit("Preprocessing images...")
                self._preprocess_images(preprocessor)
                self.progress_updated.emit(75)
            
            if 'augment' in self.operations:
                self.log_updated.emit("Augmenting dataset...")
                self._augment_dataset(augmentator)
                self.progress_updated.emit(90)
            
            self.log_updated.emit("Validating dataset structure...")
            issues = file_manager.validate_dataset_structure()
            if issues:
                for issue in issues:
                    self.log_updated.emit(f"Issue: {issue}")
            else:
                self.log_updated.emit("Dataset structure is valid")
            
            self.progress_updated.emit(100)
            self.log_updated.emit("Data processing completed successfully!")
            self.finished_signal.emit(True)
            
        except Exception as e:
            self.log_updated.emit(f"Error: {str(e)}")
            self.finished_signal.emit(False)
    
    def _preprocess_images(self, preprocessor):
        for subset in ['train', 'val', 'test']:
            subset_dir = os.path.join(self.config['paths']['data_dir'], subset, 'images')
            if os.path.exists(subset_dir):
                images = [f for f in os.listdir(subset_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                for img_file in images[:10]:
                    img_path = os.path.join(subset_dir, img_file)
                    processed = preprocessor.preprocess_single(img_path)
    
    def _augment_dataset(self, augmentator):
        pass

class DataDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.source_directory = ""
        
        self.setWindowTitle("Data Preparation")
        self.setGeometry(200, 200, 600, 500)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        source_group = QGroupBox("Source Data")
        source_layout = QVBoxLayout()
        
        source_btn_layout = QHBoxLayout()
        self.source_label = QLabel("No directory selected")
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_source)
        
        source_btn_layout.addWidget(self.source_label)
        source_btn_layout.addWidget(self.browse_btn)
        source_layout.addLayout(source_btn_layout)
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)
        
        operations_group = QGroupBox("Operations")
        operations_layout = QVBoxLayout()
        
        self.create_structure_cb = QCheckBox("Create project structure")
        self.create_structure_cb.setChecked(True)
        operations_layout.addWidget(self.create_structure_cb)
        
        self.split_dataset_cb = QCheckBox("Split dataset (train/val/test)")
        self.split_dataset_cb.setChecked(True)
        operations_layout.addWidget(self.split_dataset_cb)
        
        self.preprocess_cb = QCheckBox("Preprocess images")
        operations_layout.addWidget(self.preprocess_cb)
        
        self.augment_cb = QCheckBox("Apply data augmentation")
        operations_layout.addWidget(self.augment_cb)
        
        operations_group.setLayout(operations_layout)
        layout.addWidget(operations_group)
        
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()
        
        train_layout = QHBoxLayout()
        train_layout.addWidget(QLabel("Train ratio:"))
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.1, 0.9)
        self.train_ratio_spin.setValue(self.config['data']['train_ratio'])
        self.train_ratio_spin.setSingleStep(0.05)
        train_layout.addWidget(self.train_ratio_spin)
        params_layout.addLayout(train_layout)
        
        val_layout = QHBoxLayout()
        val_layout.addWidget(QLabel("Validation ratio:"))
        self.val_ratio_spin = QDoubleSpinBox()
        self.val_ratio_spin.setRange(0.05, 0.5)
        self.val_ratio_spin.setValue(self.config['data']['val_ratio'])
        self.val_ratio_spin.setSingleStep(0.05)
        val_layout.addWidget(self.val_ratio_spin)
        params_layout.addLayout(val_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.clicked.connect(self.start_processing)
        button_layout.addWidget(self.start_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def browse_source(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Source Data Directory"
        )
        
        if directory:
            self.source_directory = directory
            self.source_label.setText(f"Selected: {directory}")
    
    def start_processing(self):
        operations = []
        
        if self.create_structure_cb.isChecked():
            operations.append('create_structure')
        
        if self.split_dataset_cb.isChecked():
            operations.append('split_dataset')
        
        if self.preprocess_cb.isChecked():
            operations.append('preprocess')
        
        if self.augment_cb.isChecked():
            operations.append('augment')
        
        if not operations:
            self.log_text.append("No operations selected")
            return
        
        self.config['data']['train_ratio'] = self.train_ratio_spin.value()
        self.config['data']['val_ratio'] = self.val_ratio_spin.value()
        self.config['data']['test_ratio'] = 1.0 - self.train_ratio_spin.value() - self.val_ratio_spin.value()
        
        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        self.worker_thread = DataProcessingThread(
            self.config, self.source_directory, operations
        )
        
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue)
        self.worker_thread.log_updated.connect(self.log_text.append)
        self.worker_thread.finished_signal.connect(self.processing_finished)
        
        self.worker_thread.start()
    
    def processing_finished(self, success):
        self.start_btn.setEnabled(True)
        if success:
            self.log_text.append("Processing completed successfully!")
        else:
            self.log_text.append("Processing failed!")