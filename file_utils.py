import os
import shutil
import json
import csv
from pathlib import Path
import glob

class FileManager:
    def __init__(self, config):
        self.config = config
        self.data_dir = config['paths']['data_dir']
        self.model_dir = config['paths']['model_dir']
        self.results_dir = config['paths']['results_dir']
    
    def create_project_structure(self):
        directories = [
            self.data_dir,
            self.model_dir,
            self.results_dir,
            os.path.join(self.data_dir, 'raw'),
            os.path.join(self.data_dir, 'processed'),
            os.path.join(self.data_dir, 'annotations'),
            os.path.join(self.data_dir, 'train', 'images'),
            os.path.join(self.data_dir, 'train', 'labels'),
            os.path.join(self.data_dir, 'val', 'images'),
            os.path.join(self.data_dir, 'val', 'labels'),
            os.path.join(self.data_dir, 'test', 'images'),
            os.path.join(self.data_dir, 'test', 'labels'),
            os.path.join(self.results_dir, 'training_logs'),
            os.path.join(self.results_dir, 'predictions'),
            os.path.join(self.results_dir, 'analytics')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_image_files(self, directory, extensions=['.jpg', '.jpeg', '.png', '.bmp']):
        image_files = []
        for ext in extensions:
            pattern = os.path.join(directory, f'*{ext}')
            image_files.extend(glob.glob(pattern))
            pattern = os.path.join(directory, f'*{ext.upper()}')
            image_files.extend(glob.glob(pattern))
        return sorted(image_files)
    
    def get_annotation_files(self, directory, extension='.txt'):
        pattern = os.path.join(directory, f'*{extension}')
        return sorted(glob.glob(pattern))
    
    def copy_files(self, source_files, destination_dir):
        os.makedirs(destination_dir, exist_ok=True)
        
        copied_files = []
        for source_file in source_files:
            filename = os.path.basename(source_file)
            destination_file = os.path.join(destination_dir, filename)
            shutil.copy2(source_file, destination_file)
            copied_files.append(destination_file)
        
        return copied_files
    
    def move_files(self, source_files, destination_dir):
        os.makedirs(destination_dir, exist_ok=True)
        
        moved_files = []
        for source_file in source_files:
            filename = os.path.basename(source_file)
            destination_file = os.path.join(destination_dir, filename)
            shutil.move(source_file, destination_file)
            moved_files.append(destination_file)
        
        return moved_files
    
    def save_predictions(self, predictions, filename):
        output_path = os.path.join(self.results_dir, 'predictions', filename)
        
        if filename.endswith('.json'):
            with open(output_path, 'w') as f:
                json.dump(predictions, f, indent=2)
        elif filename.endswith('.csv'):
            self._save_predictions_csv(predictions, output_path)
        else:
            raise ValueError("Unsupported file format. Use .json or .csv")
        
        return output_path
    
    def _save_predictions_csv(self, predictions, filepath):
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['image_path', 'class_id', 'class_name', 'confidence', 
                         'x1', 'y1', 'x2', 'y2']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in predictions:
                image_path = result['image_path']
                for pred in result['predictions']:
                    row = {
                        'image_path': image_path,
                        'class_id': pred['class_id'],
                        'class_name': pred['class_name'],
                        'confidence': pred['confidence'],
                        'x1': pred['bbox'][0],
                        'y1': pred['bbox'][1],
                        'x2': pred['bbox'][2],
                        'y2': pred['bbox'][3]
                    }
                    writer.writerow(row)
    
    def load_predictions(self, filename):
        filepath = os.path.join(self.results_dir, 'predictions', filename)
        
        if filename.endswith('.json'):
            with open(filepath, 'r') as f:
                return json.load(f)
        elif filename.endswith('.csv'):
            return self._load_predictions_csv(filepath)
        else:
            raise ValueError("Unsupported file format. Use .json or .csv")
    
    def _load_predictions_csv(self, filepath):
        predictions = {}
        
        with open(filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                image_path = row['image_path']
                
                if image_path not in predictions:
                    predictions[image_path] = {
                        'image_path': image_path,
                        'predictions': []
                    }
                
                pred = {
                    'class_id': int(row['class_id']),
                    'class_name': row['class_name'],
                    'confidence': float(row['confidence']),
                    'bbox': [int(row['x1']), int(row['y1']), 
                            int(row['x2']), int(row['y2'])]
                }
                predictions[image_path]['predictions'].append(pred)
        
        return list(predictions.values())
    
    def save_metrics(self, metrics, filename):
        output_path = os.path.join(self.results_dir, 'analytics', filename)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return output_path
    
    def load_metrics(self, filename):
        filepath = os.path.join(self.results_dir, 'analytics', filename)
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def cleanup_old_files(self, directory, max_files=10):
        files = glob.glob(os.path.join(directory, '*'))
        files.sort(key=os.path.getmtime, reverse=True)
        
        if len(files) > max_files:
            files_to_remove = files[max_files:]
            for file_path in files_to_remove:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
    
    def get_dataset_info(self, subset='train'):
        info = {}
        
        images_dir = os.path.join(self.data_dir, subset, 'images')
        labels_dir = os.path.join(self.data_dir, subset, 'labels')
        
        if os.path.exists(images_dir):
            info['num_images'] = len(self.get_image_files(images_dir))
        else:
            info['num_images'] = 0
        
        if os.path.exists(labels_dir):
            info['num_labels'] = len(self.get_annotation_files(labels_dir))
        else:
            info['num_labels'] = 0
        
        info['subset'] = subset
        return info
    
    def validate_dataset_structure(self):
        issues = []
        
        for subset in ['train', 'val', 'test']:
            images_dir = os.path.join(self.data_dir, subset, 'images')
            labels_dir = os.path.join(self.data_dir, subset, 'labels')
            
            if not os.path.exists(images_dir):
                issues.append(f"Missing images directory: {images_dir}")
                continue
            
            if not os.path.exists(labels_dir):
                issues.append(f"Missing labels directory: {labels_dir}")
                continue
            
            image_files = self.get_image_files(images_dir)
            label_files = self.get_annotation_files(labels_dir)
            
            image_names = {Path(f).stem for f in image_files}
            label_names = {Path(f).stem for f in label_files}
            
            missing_labels = image_names - label_names
            if missing_labels:
                issues.append(f"Missing labels for images in {subset}: {missing_labels}")
            
            orphaned_labels = label_names - image_names
            if orphaned_labels:
                issues.append(f"Orphaned labels in {subset}: {orphaned_labels}")
        
        return issues
    
    def export_dataset_summary(self, output_file):
        summary = {
            'project_config': self.config,
            'dataset_structure': {},
            'validation_issues': self.validate_dataset_structure()
        }
        
        for subset in ['train', 'val', 'test']:
            summary['dataset_structure'][subset] = self.get_dataset_info(subset)
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return output_file