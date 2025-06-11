from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QTextEdit, QGroupBox, 
                             QDoubleSpinBox, QListWidget, QCheckBox, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os

from ..models.yolo_model import create_yolo_model
from ..models.predictor import YOLOPredictor
from ..utils.file_utils import FileManager

class DetectionThread(QThread):
    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    detection_finished = pyqtSignal(bool, list)
    
    def __init__(self, config, model_path, image_paths, conf_threshold, nms_threshold):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.image_paths = image_paths
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
    def run(self):
        try:
            self.log_updated.emit("Loading model...")
            model = create_yolo_model(self.config['model']['num_classes'])
            predictor = YOLOPredictor(model, self.config)
            predictor.load_model(self.model_path)
            
            predictor.set_confidence_threshold(self.conf_threshold)
            predictor.set_nms_threshold(self.nms_threshold)
            
            self.log_updated.emit(f"Processing {len(self.image_paths)} images...")
            
            results = []
            for i, image_path in enumerate(self.image_paths):
                self.log_updated.emit(f"Processing: {os.path.basename(image_path)}")
                
                predictions = predictor.predict_single_image(image_path)
                
                result = {
                    'image_path': image_path,
                    'predictions': predictions
                }
                results.append(result)
                
                progress = int(((i + 1) / len(self.image_paths)) * 100)
                self.progress_updated.emit(progress)
            
            self.log_updated.emit("Detection completed successfully!")
            self.detection_finished.emit(True, results)
            
        except Exception as e:
            self.log_updated.emit(f"Detection failed: {str(e)}")
            self.detection_finished.emit(False, [])

class DetectDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.model_path = ""
        self.image_paths = []
        self.detection_results = []
        
        self.setWindowTitle("Object Detection")
        self.setGeometry(200, 200, 700, 600)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        
        model_btn_layout = QHBoxLayout()
        self.model_label = QLabel("No model selected")
        self.browse_model_btn = QPushButton("Browse Model")
        self.browse_model_btn.clicked.connect(self.browse_model)
        
        model_btn_layout.addWidget(self.model_label)
        model_btn_layout.addWidget(self.browse_model_btn)
        model_layout.addLayout(model_btn_layout)
        
        available_models = self._get_available_models()
        if available_models:
            model_combo_layout = QHBoxLayout()
            model_combo_layout.addWidget(QLabel("Available models:"))
            self.model_combo = QComboBox()
            self.model_combo.addItems(available_models)
            self.model_combo.currentTextChanged.connect(self.select_model)
            model_combo_layout.addWidget(self.model_combo)
            model_layout.addLayout(model_combo_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        images_group = QGroupBox("Input Images")
        images_layout = QVBoxLayout()
        
        images_btn_layout = QHBoxLayout()
        self.add_images_btn = QPushButton("Add Images")
        self.add_images_btn.clicked.connect(self.add_images)
        
        self.add_folder_btn = QPushButton("Add Folder")
        self.add_folder_btn.clicked.connect(self.add_folder)
        
        self.clear_images_btn = QPushButton("Clear All")
        self.clear_images_btn.clicked.connect(self.clear_images)
        
        images_btn_layout.addWidget(self.add_images_btn)
        images_btn_layout.addWidget(self.add_folder_btn)
        images_btn_layout.addWidget(self.clear_images_btn)
        images_layout.addLayout(images_btn_layout)
        
        self.images_list = QListWidget()
        self.images_list.setMaximumHeight(150)
        images_layout.addWidget(self.images_list)
        
        images_group.setLayout(images_layout)
        layout.addWidget(images_group)
        
        params_group = QGroupBox("Detection Parameters")
        params_layout = QVBoxLayout()
        
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence Threshold:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 1.0)
        self.conf_spin.setValue(0.5)
        self.conf_spin.setSingleStep(0.05)
        conf_layout.addWidget(self.conf_spin)
        params_layout.addLayout(conf_layout)
        
        nms_layout = QHBoxLayout()
        nms_layout.addWidget(QLabel("NMS Threshold:"))
        self.nms_spin = QDoubleSpinBox()
        self.nms_spin.setRange(0.1, 1.0)
        self.nms_spin.setValue(0.4)
        self.nms_spin.setSingleStep(0.05)
        nms_layout.addWidget(self.nms_spin)
        params_layout.addLayout(nms_layout)
        
        self.save_results_cb = QCheckBox("Save results to file")
        self.save_results_cb.setChecked(True)
        params_layout.addWidget(self.save_results_cb)
        
        self.save_images_cb = QCheckBox("Save annotated images")
        params_layout.addWidget(self.save_images_cb)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        button_layout = QHBoxLayout()
        
        self.detect_btn = QPushButton("Start Detection")
        self.detect_btn.clicked.connect(self.start_detection)
        button_layout.addWidget(self.detect_btn)
        
        self.save_btn = QPushButton("Save Results")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_results)
        button_layout.addWidget(self.save_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _get_available_models(self):
        model_dir = self.config['paths']['model_dir']
        if os.path.exists(model_dir):
            models = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            return models
        return []
    
    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", 
            self.config['paths']['model_dir'],
            "PyTorch Models (*.pth);;All Files (*)"
        )
        
        if file_path:
            self.model_path = file_path
            self.model_label.setText(f"Selected: {os.path.basename(file_path)}")
    
    def select_model(self, model_name):
        if model_name:
            self.model_path = os.path.join(self.config['paths']['model_dir'], model_name)
            self.model_label.setText(f"Selected: {model_name}")
    
    def add_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        
        for file_path in file_paths:
            if file_path not in self.image_paths:
                self.image_paths.append(file_path)
                self.images_list.addItem(os.path.basename(file_path))
    
    def add_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Image Folder"
        )
        
        if folder_path:
            file_manager = FileManager(self.config)
            image_files = file_manager.get_image_files(folder_path)
            
            for file_path in image_files:
                if file_path not in self.image_paths:
                    self.image_paths.append(file_path)
                    self.images_list.addItem(os.path.basename(file_path))
    
    def clear_images(self):
        self.image_paths.clear()
        self.images_list.clear()
    
    def start_detection(self):
        if not self.model_path:
            self.log_text.append("Please select a model first")
            return
        
        if not self.image_paths:
            self.log_text.append("Please add images for detection")
            return
        
        self.detect_btn.setEnabled(False)
        self.log_text.clear()
        
        self.detection_thread = DetectionThread(
            self.config,
            self.model_path,
            self.image_paths,
            self.conf_spin.value(),
            self.nms_spin.value()
        )
        
        self.detection_thread.log_updated.connect(self.log_text.append)
        self.detection_thread.detection_finished.connect(self.detection_finished)
        
        self.detection_thread.start()
    
    def detection_finished(self, success, results):
        self.detect_btn.setEnabled(True)
        
        if success:
            self.detection_results = results
            self.save_btn.setEnabled(True)
            
            total_detections = sum(len(r['predictions']) for r in results)
            self.log_text.append(f"Detection completed! Found {total_detections} objects")
            
            if self.save_results_cb.isChecked():
                self.save_results()
        else:
            self.log_text.append("Detection failed!")
    
    def save_results(self):
        if not self.detection_results:
            self.log_text.append("No results to save")
            return
        
        file_manager = FileManager(self.config)
        
        filename = "detection_results.json"
        saved_path = file_manager.save_predictions(self.detection_results, filename)
        self.log_text.append(f"Results saved to: {saved_path}")
        
        if self.save_images_cb.isChecked():
            from ..utils.visualization import Visualizer
            visualizer = Visualizer(self.config)
            
            output_dir = os.path.join(self.config['paths']['results_dir'], 'annotated_images')
            os.makedirs(output_dir, exist_ok=True)
            
            for result in self.detection_results:
                image_path = result['image_path']
                predictions = result['predictions']
                
                output_path = os.path.join(output_dir, f"annotated_{os.path.basename(image_path)}")
                visualizer.visualize_predictions(image_path, predictions, output_path)
            
            self.log_text.append(f"Annotated images saved to: {output_dir}")