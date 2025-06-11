#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
from tqdm import tqdm

from src.utils.config_manager import ConfigManager
from src.models.yolo_model import create_yolo_model
from src.models.predictor import YOLOPredictor
from src.utils.file_utils import FileManager
from src.utils.visualization import Visualizer

def detect_objects_in_images(config, model_path, image_paths, conf_threshold, nms_threshold, save_results=True, save_images=False):
    model = create_yolo_model(config['model']['num_classes'])
    predictor = YOLOPredictor(model, config)
    predictor.load_model(model_path)
    
    predictor.set_confidence_threshold(conf_threshold)
    predictor.set_nms_threshold(nms_threshold)
    
    file_manager = FileManager(config)
    visualizer = Visualizer(config)
    
    print(f"Processing {len(image_paths)} images...")
    
    results = []
    total_detections = 0
    
    for image_path in tqdm(image_paths, desc="Detecting objects"):
        predictions = predictor.predict_single_image(image_path)
        
        result = {
            'image_path': image_path,
            'predictions': predictions
        }
        results.append(result)
        
        total_detections += len(predictions)
        
        if save_images and predictions:
            output_dir = os.path.join(config['paths']['results_dir'], 'annotated_images')
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
            visualizer.visualize_predictions(image_path, predictions, output_path)
    
    print(f"\nDetection completed!")
    print(f"Total images processed: {len(image_paths)}")
    print(f"Total objects detected: {total_detections}")
    
    class_counts = {}
    for result in results:
        for pred in result['predictions']:
            class_name = pred['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    if class_counts:
        print("\nDetections by class:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")
    
    if save_results:
        output_file = file_manager.save_predictions(results, "detection_results.json")
        print(f"\nResults saved to: {output_file}")
        
        csv_file = file_manager.save_predictions(results, "detection_results.csv")
        print(f"CSV results saved to: {csv_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Detect objects in images using trained YOLO model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to image file or directory')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--nms-threshold', type=float, default=0.4,
                       help='NMS threshold for detections')
    parser.add_argument('--save-images', action='store_true',
                       help='Save annotated images')
    parser.add_argument('--no-save-results', action='store_true',
                       help='Do not save detection results to file')
    parser.add_argument('--output-dir', type=str,
                       help='Custom output directory for results')
    
    args = parser.parse_args()
    
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    
    if args.output_dir:
        config['paths']['results_dir'] = args.output_dir
    
    print("Maritime Object Detection - Detection Script")
    print(f"Configuration: {args.config}")
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"NMS threshold: {args.nms_threshold}")
    print()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.input):
        print(f"Error: Input path not found: {args.input}")
        return
    
    image_paths = []
    
    if os.path.isfile(args.input):
        if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_paths = [args.input]
        else:
            print(f"Error: Input file is not a supported image format")
            return
    elif os.path.isdir(args.input):
        file_manager = FileManager(config)
        image_paths = file_manager.get_image_files(args.input)
        
        if not image_paths:
            print(f"Error: No image files found in directory: {args.input}")
            return
    else:
        print(f"Error: Input path is neither a file nor a directory: {args.input}")
        return
    
    results = detect_objects_in_images(
        config=config,
        model_path=args.model,
        image_paths=image_paths,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        save_results=not args.no_save_results,
        save_images=args.save_images
    )
    
    if args.save_images:
        output_dir = os.path.join(config['paths']['results_dir'], 'annotated_images')
        print(f"Annotated images saved to: {output_dir}")
    
    print("\nDetection completed successfully!")

if __name__ == '__main__':
    main()