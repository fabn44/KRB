import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch

class MetricsCalculator:
    def __init__(self, num_classes, class_names):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        self.true_positives = np.zeros(self.num_classes)
        self.false_positives = np.zeros(self.num_classes)
        self.false_negatives = np.zeros(self.num_classes)
        self.predictions = []
        self.ground_truths = []
    
    def calculate_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def update(self, predictions, ground_truths, iou_threshold=0.5):
        for pred_class in range(self.num_classes):
            pred_boxes = [p for p in predictions if p['class_id'] == pred_class]
            gt_boxes = [g for g in ground_truths if g['class_id'] == pred_class]
            
            matched_gt = set()
            
            for pred in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold:
                    self.true_positives[pred_class] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    self.false_positives[pred_class] += 1
            
            self.false_negatives[pred_class] += len(gt_boxes) - len(matched_gt)
    
    def calculate_precision(self, class_id=None):
        if class_id is not None:
            tp = self.true_positives[class_id]
            fp = self.false_positives[class_id]
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        precisions = []
        for i in range(self.num_classes):
            precisions.append(self.calculate_precision(i))
        return np.mean(precisions)
    
    def calculate_recall(self, class_id=None):
        if class_id is not None:
            tp = self.true_positives[class_id]
            fn = self.false_negatives[class_id]
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        recalls = []
        for i in range(self.num_classes):
            recalls.append(self.calculate_recall(i))
        return np.mean(recalls)
    
    def calculate_f1_score(self, class_id=None):
        if class_id is not None:
            precision = self.calculate_precision(class_id)
            recall = self.calculate_recall(class_id)
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        f1_scores = []
        for i in range(self.num_classes):
            f1_scores.append(self.calculate_f1_score(i))
        return np.mean(f1_scores)
    
    def calculate_map(self, iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
        aps = []
        
        for threshold in iou_thresholds:
            class_aps = []
            for class_id in range(self.num_classes):
                precision = self.calculate_precision(class_id)
                class_aps.append(precision)
            aps.append(np.mean(class_aps))
        
        return np.mean(aps)
    
    def get_detailed_metrics(self):
        metrics = {
            'overall_precision': self.calculate_precision(),
            'overall_recall': self.calculate_recall(),
            'overall_f1': self.calculate_f1_score(),
            'map': self.calculate_map()
        }
        
        for i in range(self.num_classes):
            class_name = self.class_names[i]
            metrics[f'precision_{class_name}'] = self.calculate_precision(i)
            metrics[f'recall_{class_name}'] = self.calculate_recall(i)
            metrics[f'f1_{class_name}'] = self.calculate_f1_score(i)
            metrics[f'tp_{class_name}'] = int(self.true_positives[i])
            metrics[f'fp_{class_name}'] = int(self.false_positives[i])
            metrics[f'fn_{class_name}'] = int(self.false_negatives[i])
        
        return metrics
    
    def print_metrics(self):
        metrics = self.get_detailed_metrics()
        
        print("Overall Metrics:")
        print(f"  Precision: {metrics['overall_precision']:.4f}")
        print(f"  Recall: {metrics['overall_recall']:.4f}")
        print(f"  F1-Score: {metrics['overall_f1']:.4f}")
        print(f"  mAP: {metrics['map']:.4f}")
        print()
        
        print("Per-Class Metrics:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}:")
            print(f"    Precision: {metrics[f'precision_{class_name}']:.4f}")
            print(f"    Recall: {metrics[f'recall_{class_name}']:.4f}")
            print(f"    F1-Score: {metrics[f'f1_{class_name}']:.4f}")
            print(f"    TP: {metrics[f'tp_{class_name}']}, FP: {metrics[f'fp_{class_name}']}, FN: {metrics[f'fn_{class_name}']}")
            print()

class PerformanceMonitor:
    def __init__(self):
        self.training_metrics = []
        self.validation_metrics = []
        self.inference_times = []
        
    def log_training_metrics(self, epoch, loss, learning_rate):
        self.training_metrics.append({
            'epoch': epoch,
            'loss': loss,
            'learning_rate': learning_rate
        })
    
    def log_validation_metrics(self, epoch, loss, metrics):
        entry = {
            'epoch': epoch,
            'loss': loss,
            **metrics
        }
        self.validation_metrics.append(entry)
    
    def log_inference_time(self, time_ms, image_size):
        self.inference_times.append({
            'time_ms': time_ms,
            'image_size': image_size,
            'fps': 1000 / time_ms if time_ms > 0 else 0
        })
    
    def get_best_epoch(self, metric='overall_f1'):
        if not self.validation_metrics:
            return None
        
        best_idx = 0
        best_value = self.validation_metrics[0].get(metric, 0)
        
        for i, entry in enumerate(self.validation_metrics):
            value = entry.get(metric, 0)
            if value > best_value:
                best_value = value
                best_idx = i
        
        return self.validation_metrics[best_idx]['epoch']
    
    def get_average_fps(self):
        if not self.inference_times:
            return 0
        
        total_fps = sum(entry['fps'] for entry in self.inference_times)
        return total_fps / len(self.inference_times)
    
    def export_metrics(self, filepath):
        import json
        
        data = {
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'inference_times': self.inference_times,
            'summary': {
                'best_epoch': self.get_best_epoch(),
                'average_fps': self.get_average_fps()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)