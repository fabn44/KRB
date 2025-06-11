import yaml
import os
import json

class ConfigManager:
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()
    
    def save_config(self, config=None):
        if config is None:
            config = self.config
            
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def get_parameter(self, key_path):
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def set_parameter(self, key_path, value):
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        self.save_config()
    
    def update_training_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.config['training']:
                self.config['training'][key] = value
        self.save_config()
    
    def update_model_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.config['model']:
                self.config['model'][key] = value
        self.save_config()
    
    def create_directories(self):
        dirs_to_create = [
            self.config['paths']['data_dir'],
            self.config['paths']['model_dir'],
            self.config['paths']['results_dir'],
            os.path.join(self.config['paths']['data_dir'], 'raw'),
            os.path.join(self.config['paths']['data_dir'], 'processed'),
            os.path.join(self.config['paths']['data_dir'], 'annotations'),
            os.path.join(self.config['paths']['results_dir'], 'training_logs'),
            os.path.join(self.config['paths']['results_dir'], 'predictions'),
            os.path.join(self.config['paths']['results_dir'], 'analytics')
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
    
    def export_config(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def import_config(self, filepath):
        with open(filepath, 'r') as f:
            imported_config = json.load(f)
        
        self.config.update(imported_config)
        self.save_config()
    
    def reset_to_default(self):
        self.config = self._get_default_config()
        self.save_config()
    
    def _get_default_config(self):
        return {
            'model': {
                'architecture': 'yolov5',
                'input_size': [640, 640],
                'num_classes': 5,
                'classes': ['dock', 'boat', 'boat_lift', 'jetski', 'car']
            },
            'training': {
                'batch_size': 16,
                'epochs': 100,
                'learning_rate': 0.001,
                'optimizer': 'adam'
            },
            'data': {
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15
            },
            'paths': {
                'data_dir': 'data/',
                'model_dir': 'data/models/',
                'results_dir': 'results/'
            }
        }
    
    def get_class_names(self):
        return self.config['model']['classes']
    
    def get_num_classes(self):
        return self.config['model']['num_classes']
    
    def validate_config(self):
        required_keys = [
            'model.architecture',
            'model.input_size',
            'model.num_classes',
            'training.batch_size',
            'training.epochs',
            'paths.data_dir'
        ]
        
        for key in required_keys:
            if self.get_parameter(key) is None:
                return False, f"Missing required parameter: {key}"
        
        if len(self.config['model']['classes']) != self.config['model']['num_classes']:
            return False, "Number of classes doesn't match class names list"
        
        return True, "Configuration is valid"