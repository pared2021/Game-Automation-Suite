import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from utils.logger import detailed_logger
from utils.config_manager import config_manager

class EnhancedImageRecognition:
    def __init__(self):
        self.logger = detailed_logger
        self.config = config_manager.get('image_recognition', {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载预训练的ResNet模型
        self.model = resnet50(pretrained=True).to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    async def analyze_scene(self, image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # 获取前5个最可能的类别
        _, indices = torch.sort(output, descending=True)
        indices = indices[:, :5]
        
        # 获取类别名称
        with open('imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]
        
        results = [classes[idx] for idx in indices[0]]
        self.logger.info(f"Scene analysis results: {results}")
        return results

    async def detect_objects(self, image):
        # 使用YOLOv5进行对象检测
        results = self.yolo_model(image)
        objects = results.pandas().xyxy[0].to_dict(orient="records")
        self.logger.info(f"Detected objects: {objects}")
        return objects

    async def recognize_text(self, image):
        # 使用Tesseract OCR进行文本识别
        text = pytesseract.image_to_string(image)
        self.logger.info(f"Recognized text: {text}")
        return text

    async def analyze_color_scheme(self, image):
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
        # 使用Canny边缘检测
        edges = cv2.Canny(image, 100, 200)
        self.logger.info("Edge detection completed")
        return edges

    async def segment_image(self, image):
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
