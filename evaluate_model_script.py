#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import argparse
import time

from src.utils.config_manager import ConfigManager
from src.models.yolo_model import create_yolo_model
from src.models.predictor import YOLOPredictor
from src.data.data_loader import MaritimeDataset
from src.data.augmentation import DataAugmentation
from src.utils.metrics import MetricsCalculator, PerformanceMonitor
from src.utils.visualization import Visualizer

def evaluate_model(config, model_path, subset='test'):
    model = create_yolo_model(config['model']['num_classes'])
    predictor = YOLOPredictor(model, config)
    predictor.load_model(model_path)
    
    augmentation = DataAugmentation(config)
    
    dataset = MaritimeDataset(
        f"{config['paths']['data_dir']}/{subset}/images",
        f"{config['paths']['data_dir']}/{subset}/labels",
        transform=augmentation.get_validation_transform()
    )
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    metrics_calc = MetricsCalculator(
        config['model']['num_classes'],
        config['model']['classes']
    )
    
    performance_monitor = PerformanceMonitor()
    
    print(f"Evaluating on {len(dataset)} images...")
    
    total_time = 0
    processed_images = 0
    
    for i, sample in enumerate(data_loader):
        image_name = sample['image_name'][0]
        image_path = f"{config['paths']['data_dir']}/{subset}/images/{image_name}"
        
        start_time = time.time()
        predictions = predictor.predict_single_image(image_path)
        inference_time = (time.time() - start_time) * 1000
        
        total_time += inference_time
        processed_images += 1
        
        performance_monitor.log_inference_time(
            inference_time, 
            config['model']['input_size']
        )
        
        gt_boxes = sample['boxes'][0].numpy()
        gt_labels = sample['labels'][0].numpy()
        
        ground_truths = []
        for box, label in zip(gt_boxes, gt_labels):
            if len(box) == 4:
                x_center, y_center, width, height = box
                x1 = int((x_center - width/2) * config['model']['input_size'][0])
                y1 = int((y_center - height/2) * config['model']['input_size'][1])
                x2 = int((x_center + width/2) * config['model']['input_size'][0])
                y2 = int((y_center + height/2) * config['model']['input_size'][1])
                
                ground_truths.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_id': int(label)
                })
        
        metrics_calc.update(predictions, ground_truths)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(dataset)} images")
    
    metrics = metrics_calc.get_detailed_metrics()
    
    print("\nEvaluation Results:")
    print("=" * 50)
    print(f"Overall Precision: {metrics['overall_precision']:.4f}")
    print(f"Overall Recall: {metrics['overall_recall']:.4f}")
    print(f"Overall F1-Score: {metrics['overall_f1']:.4f}")
    print(f"mAP: {metrics['map']:.4f}")
    print()
    
    print("Per-Class Results:")
    print("-" * 30)
    for class_name in config['model']['classes']:
        precision = metrics.get(f'precision_{class_name}', 0)
        recall = metrics.get(f'recall_{class_name}', 0)
        f1 = metrics.get(f'f1_{class_name}', 0)
        
        print(f"{class_name:>12}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    print()
    print("Performance Metrics:")
    print("-" * 30)
    avg_time = total_time / processed_images
    avg_fps = performance_monitor.get_average_fps()
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Average FPS: {avg_fps:.2f}")
    
    return metrics, performance_monitor

def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO model for maritime object detection')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--subset', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Dataset subset to evaluate on')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    parser.add_argument('--nms-threshold', type=float, default=0.4,
                       help='NMS threshold for predictions')
    parser.add_argument('--save-visualizations', action='store_true',
                       help='Save visualization plots')
    
    args = parser.parse_args()
    
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    
    print("Maritime Object Detection - Evaluation Script")
    print(f"Configuration: {args.config}")
    print(f"Model: {args.model}")
    print(f"Subset: {args.subset}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"NMS threshold: {args.nms_threshold}")
    print()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    metrics, performance_monitor = evaluate_model(config, args.model, args.subset)
    
    results_dir = os.path.join(config['paths']['results_dir'], 'evaluation')
    os.makedirs(results_dir, exist_ok=True)
    
    metrics_file = os.path.join(results_dir, f'metrics_{args.subset}.json')
    performance_file = os.path.join(results_dir, f'performance_{args.subset}.json')
    
    import json
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    performance_monitor.export_metrics(performance_file)
    
    print(f"\nResults saved to:")
    print(f"  Metrics: {metrics_file}")
    print(f"  Performance: {performance_file}")
    
    if args.save_visualizations:
        visualizer = Visualizer(config)
        
        class_distribution = {}
        for class_name in config['model']['classes']:
            tp_key = f'tp_{class_name}'
            if tp_key in metrics:
                class_distribution[class_name] = metrics[tp_key]
        
        viz_dir = os.path.join(results_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        visualizer.plot_class_distribution(
            class_distribution, 
            os.path.join(viz_dir, 'class_distribution.png')
        )
        
        visualizer.plot_metrics_comparison(
            metrics,
            os.path.join(viz_dir, 'metrics_comparison.png')
        )
        
        print(f"  Visualizations: {viz_dir}")

if __name__ == '__main__':
    main()