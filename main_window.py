import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QPushButton, QWidget, QTextEdit, QStatusBar, QMenuBar,
                             QAction, QMessageBox, QTabWidget, QLabel)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

from .data_dialog import DataDialog
from .train_dialog import TrainDialog
from .detect_dialog import DetectDialog
from .analytics_dialog import AnalyticsDialog
from ..utils.config_manager import ConfigManager

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config
        
        self.setWindowTitle("Maritime Object Detection System")
        self.setGeometry(100, 100, 1200, 800)
        
        self.init_ui()
        self.init_menu()
        self.init_statusbar()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        title_label = QLabel("Maritime Object Detection System")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        button_layout = QHBoxLayout()
        
        self.data_btn = QPushButton("Data Preparation")
        self.data_btn.setMinimumHeight(60)
        self.data_btn.clicked.connect(self.open_data_dialog)
        button_layout.addWidget(self.data_btn)
        
        self.train_btn = QPushButton("Model Training")
        self.train_btn.setMinimumHeight(60)
        self.train_btn.clicked.connect(self.open_train_dialog)
        button_layout.addWidget(self.train_btn)
        
        self.detect_btn = QPushButton("Object Detection")
        self.detect_btn.setMinimumHeight(60)
        self.detect_btn.clicked.connect(self.open_detect_dialog)
        button_layout.addWidget(self.detect_btn)
        
        self.analytics_btn = QPushButton("Analytics & Reports")
        self.analytics_btn.setMinimumHeight(60)
        self.analytics_btn.clicked.connect(self.open_analytics_dialog)
        button_layout.addWidget(self.analytics_btn)
        
        main_layout.addLayout(button_layout)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text)
        
        info_layout = QHBoxLayout()
        
        config_info = QLabel(f"Model: {self.config['model']['architecture']} | "
                            f"Classes: {self.config['model']['num_classes']} | "
                            f"Input Size: {self.config['model']['input_size']}")
        info_layout.addWidget(config_info)
        
        main_layout.addLayout(info_layout)
        
    def init_menu(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('File')
        
        new_action = QAction('New Project', self)
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        
        open_action = QAction('Open Project', self)
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)
        
        save_action = QAction('Save Project', self)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        tools_menu = menubar.addMenu('Tools')
        
        config_action = QAction('Configuration', self)
        config_action.triggered.connect(self.open_config)
        tools_menu.addAction(config_action)
        
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def init_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    def open_data_dialog(self):
        dialog = DataDialog(self.config, self)
        dialog.exec_()
        
    def open_train_dialog(self):
        dialog = TrainDialog(self.config, self)
        dialog.exec_()
        
    def open_detect_dialog(self):
        dialog = DetectDialog(self.config, self)
        dialog.exec_()
        
    def open_analytics_dialog(self):
        dialog = AnalyticsDialog(self.config, self)
        dialog.exec_()
        
    def new_project(self):
        self.config_manager.reset_to_default()
        self.config = self.config_manager.config
        self.config_manager.create_directories()
        self.log_message("New project created")
        
    def open_project(self):
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project Configuration", 
            "", "YAML files (*.yaml);;All Files (*)"
        )
        
        if file_path:
            self.config_manager = ConfigManager(file_path)
            self.config = self.config_manager.config
            self.log_message(f"Project opened: {file_path}")
            
    def save_project(self):
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Project Configuration", 
            "config.yaml", "YAML files (*.yaml);;All Files (*)"
        )
        
        if file_path:
            self.config_manager.config_path = file_path
            self.config_manager.save_config()
            self.log_message(f"Project saved: {file_path}")
            
    def open_config(self):
        msg = QMessageBox()
        msg.setWindowTitle("Configuration")
        msg.setText("Configuration editing not implemented yet")
        msg.exec_()
        
    def show_about(self):
        msg = QMessageBox()
        msg.setWindowTitle("About")
        msg.setText("Maritime Object Detection System\n"
                   "Version 1.0\n"
                   "Built with PyQt5 and PyTorch")
        msg.exec_()
        
    def log_message(self, message):
        self.log_text.append(f"[INFO] {message}")
        self.status_bar.showMessage(message)
        
    def log_error(self, message):
        self.log_text.append(f"[ERROR] {message}")
        self.status_bar.showMessage(f"Error: {message}")
        
    def log_warning(self, message):
        self.log_text.append(f"[WARNING] {message}")
        self.status_bar.showMessage(f"Warning: {message}")
        
    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, 'Exit Application',
            'Are you sure you want to exit?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

def main():
    app = QApplication(sys.argv)
    
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())