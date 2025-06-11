import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision.ops import nms
import os

class YOLOPredictor:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        self.input_size = tuple(config['model']['input_size'])
        self.num_classes = config['model']['num_classes']
        self.class_names = config['model']['classes']
        
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        
    def preprocess_image(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_shape = image.shape[:2]
        image_resized = cv2.resize(image, self.input_size)
        
        image_tensor = torch.from_numpy(image_resized).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor, original_shape
    
    def postprocess_outputs(self, outputs, original_shape):
        predictions = []
        
        for output in outputs:
            batch_size, _, grid_h, grid_w = output.shape
            output = output.view(batch_size, 3, 5 + self.num_classes, grid_h, grid_w)
            output = output.permute(0, 1, 3, 4, 2).contiguous()
            
            for b in range(batch_size):
                for a in range(3):
                    for h in range(grid_h):
                        for w in range(grid_w):
                            prediction = output[b, a, h, w]
                            
                            obj_conf = torch.sigmoid(prediction[4]).item()
                            if obj_conf > self.conf_threshold:
                                
                                x = (torch.sigmoid(prediction[0]) + w) / grid_w
                                y = (torch.sigmoid(prediction[1]) + h) / grid_h
                                width = torch.sigmoid(prediction[2])
                                height = torch.sigmoid(prediction[3])
                                
                                class_probs = torch.softmax(prediction[5:], dim=0)
                                class_conf, class_pred = torch.max(class_probs, dim=0)
                                
                                final_conf = obj_conf * class_conf.item()
                                
                                if final_conf > self.conf_threshold:
                                    predictions.append([
                                        x.item(), y.item(), width.item(), height.item(),
                                        final_conf, class_pred.item()
                                    ])
        
        return self._apply_nms(predictions, original_shape)
    
    def _apply_nms(self, predictions, original_shape):
        if not predictions:
            return []
        
        predictions = np.array(predictions)
        
        boxes = predictions[:, :4]
        confidences = predictions[:, 4]
        class_ids = predictions[:, 5].astype(int)
        
        boxes[:, 0] *= original_shape[1]
        boxes[:, 1] *= original_shape[0]
        boxes[:, 2] *= original_shape[1]
        boxes[:, 3] *= original_shape[0]
        
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        boxes_tensor = torch.tensor(np.column_stack([x1, y1, x2, y2]), dtype=torch.float32)
        confidences_tensor = torch.tensor(confidences, dtype=torch.float32)
        
        keep_indices = nms(boxes_tensor, confidences_tensor, self.nms_threshold)
        
        final_predictions = []
        for idx in keep_indices:
            i = idx.item()
            final_predictions.append({
                'bbox': [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
                'confidence': confidences[i],
                'class_id': class_ids[i],
                'class_name': self.class_names[class_ids[i]]
            })
        
        return final_predictions
    
    def predict_single_image(self, image_path):
        image_tensor, original_shape = self.preprocess_image(image_path)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        predictions = self.postprocess_outputs(outputs, original_shape)
        return predictions
    
    def predict_batch(self, image_paths):
        results = []
        for image_path in image_paths:
            predictions = self.predict_single_image(image_path)
            results.append({
                'image_path': image_path,
                'predictions': predictions
            })
        return results
    
    def draw_predictions(self, image, predictions):
        if isinstance(image, str):
            image = cv2.imread(image)
        
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255)
        ]
        
        for pred in predictions:
            bbox = pred['bbox']
            confidence = pred['confidence']
            class_name = pred['class_name']
            color = colors[pred['class_id'] % len(colors)]
            
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
        
        return image
    
    def set_confidence_threshold(self, threshold):
        self.conf_threshold = threshold
    
    def set_nms_threshold(self, threshold):
        self.nms_threshold = threshold
    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()