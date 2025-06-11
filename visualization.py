import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2
import os

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.class_names = config['model']['classes']
        self.colors = self._generate_colors()
        
    def _generate_colors(self):
        np.random.seed(42)
        colors = []
        for i in range(len(self.class_names)):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors.append(color)
        return colors
    
    def plot_training_history(self, train_losses, val_losses, save_path=None):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.title('Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_class_distribution(self, class_counts, save_path=None):
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(classes)), counts, color=self.colors[:len(classes)])
        plt.title('Class Distribution')
        plt.xlabel('Classes')
        plt.ylabel('Number of Instances')
        plt.xticks(range(len(classes)), [self.class_names[i] for i in classes], rotation=45)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_metrics_comparison(self, metrics_dict, save_path=None):
        classes = list(range(len(self.class_names)))
        precision = [metrics_dict.get(f'precision_class_{i}', 0) for i in classes]
        recall = [metrics_dict.get(f'recall_class_{i}', 0) for i in classes]
        f1_score = [metrics_dict.get(f'f1_class_{i}', 0) for i in classes]
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
        plt.bar(x, recall, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Metrics Comparison by Class')
        plt.xticks(x, self.class_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def create_detection_heatmap(self, detection_results, image_shape, save_path=None):
        heatmap = np.zeros(image_shape[:2])
        
        for result in detection_results:
            for pred in result['predictions']:
                bbox = pred['bbox']
                x1, y1, x2, y2 = bbox
                heatmap[y1:y2, x1:x2] += pred['confidence']
        
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Detection Confidence')
        plt.title('Detection Heatmap')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def visualize_predictions(self, image, predictions, save_path=None):
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for pred in predictions:
            bbox = pred['bbox']
            confidence = pred['confidence']
            class_id = pred['class_id']
            class_name = pred['class_name']
            
            color = self.colors[class_id % len(self.colors)]
            
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            cv2.rectangle(
                image,
                (bbox[0], bbox[1] - label_size[1] - 10),
                (bbox[0] + label_size[0], bbox[1]),
                color,
                -1
            )
            
            cv2.putText(
                image, label,
                (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2
            )
        
        if save_path:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, image_bgr)
        
        return image
    
    def create_summary_report(self, metrics, training_history, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
        self.plot_training_history(
            training_history['train_losses'],
            training_history['val_losses'],
            os.path.join(save_dir, 'training_history.png')
        )
        
        if 'confusion_matrix' in metrics:
            y_true = metrics['y_true']
            y_pred = metrics['y_pred']
            self.plot_confusion_matrix(y_true, y_pred, 
                                     os.path.join(save_dir, 'confusion_matrix.png'))
        
        if 'class_distribution' in metrics:
            self.plot_class_distribution(metrics['class_distribution'],
                                        os.path.join(save_dir, 'class_distribution.png'))
        
        self.plot_metrics_comparison(metrics, 
                                   os.path.join(save_dir, 'metrics_comparison.png'))
    
    def save_detection_grid(self, images_with_predictions, grid_size=(2, 3), save_path=None):
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 10))
        axes = axes.flatten() if grid_size[0] * grid_size[1] > 1 else [axes]
        
        for i, (image, predictions) in enumerate(images_with_predictions[:len(axes)]):
            if isinstance(image, str):
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            annotated_image = self.visualize_predictions(image.copy(), predictions)
            
            axes[i].imshow(annotated_image)
            axes[i].set_title(f'Detection Result {i+1}')
            axes[i].axis('off')
        
        for i in range(len(images_with_predictions), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()