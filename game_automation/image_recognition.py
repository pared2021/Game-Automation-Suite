import cv2
import numpy as np
import torch
import os
import pytesseract
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from utils.logger import detailed_logger
from utils.config_manager import config_manager
from utils.error_handler import GameAutomationError

class EnhancedImageRecognition:
    def __init__(self):
        self.logger = detailed_logger
        self.config = config_manager.get('image_recognition', {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载预训练的ResNet模型
        self.model = resnet50(weights='IMAGENET1K_V1').to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    async def analyze_scene(self, image):
        """分析场景并返回前5个最可能的类别"""
        if image is None:
            raise ValueError("Image cannot be None")
            
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # 获取前5个最可能的类别
        _, indices = torch.sort(output, descending=True)
        indices = indices[:, :5]
        
        # 获取类别名称
        classes_path = os.path.join(os.path.dirname(__file__), 'resources', 'imagenet_classes.txt')
        try:
            with open(classes_path) as f:
                classes = [line.strip() for line in f.readlines()]
            
            # 确保索引在有效范围内
            valid_indices = [idx.item() for idx in indices[0] if idx.item() < len(classes)]
            results = [classes[idx] for idx in valid_indices]
            
            if not results:  # 如果没有有效结果，返回默认值
                results = ['unknown']
                
        except Exception as e:
            self.logger.error(f"Error reading classes file: {e}")
            results = ['unknown']
        
        self.logger.info(f"Scene analysis results: {results}")
        return results

    async def recognize_text(self, image):
        """使用Tesseract OCR识别图像中的文本"""
        if image is None:
            raise ValueError("Image cannot be None")
        
        # 使用Tesseract OCR进行文本识别
        text = pytesseract.image_to_string(image)
        self.logger.info(f"Recognized text: {text}")
        return text

    async def detect_objects(self, image):
        """使用OpenCV检测图像中的对象"""
        if image is None:
            raise ValueError("Image cannot be None")
            
        # 使用OpenCV的级联分类器进行对象检测
        objects = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测常见游戏元素（这里使用简化的示例）
        contours, _ = cv2.findContours(cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1], 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 20:  # 过滤掉太小的对象
                objects.append({
                    'type': 'unknown',
                    'confidence': 0.9,
                    'bbox': (x, y, w, h)
                })
        
        self.logger.info(f"Detected objects: {objects}")
        return objects

    async def analyze_color_scheme(self, image):
        """分析图像的主要颜色方案"""
        if image is None:
            raise ValueError("Image cannot be None")
            
        # 分析图像的主要颜色方案
        pixels = np.float32(image.reshape(-1, 3))
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        dominant_color = palette[np.argmax(counts)]
        self.logger.info(f"Dominant color: {dominant_color}")
        return dominant_color.tolist()

    async def detect_edges(self, image):
        """使用Canny算法进行边缘检测"""
        if image is None:
            raise ValueError("Image cannot be None")
            
        # 使用Canny边缘检测
        edges = cv2.Canny(image, 100, 200)
        self.logger.info("Edge detection completed")
        return edges

    async def segment_image(self, image):
        """使用分水岭算法进行图像分割"""
        if image is None:
            raise ValueError("Image cannot be None")
            
        # 使用分水岭算法进行图像分割
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(image, markers)
        self.logger.info("Image segmentation completed")
        return markers

enhanced_image_recognition = EnhancedImageRecognition()
