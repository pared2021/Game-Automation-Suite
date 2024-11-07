import cv2
import numpy as np
from paddleocr import PaddleOCR
from utils.config_manager import config_manager
from utils.error_handler import log_exception
from utils.logger import detailed_logger
import enchant
import re
from collections import defaultdict

class OCRUtils:
    def __init__(self):
        self.logger = detailed_logger
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        self.config = config_manager.get('ocr', {})
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.default_dpi_scale = self.config.get('default_dpi_scale', 1.0)
        self.spell_checker = enchant.Dict("en_US")
        self.context_words = set(self.config.get('context_words', []))
        self.adaptive_learning = self.config.get('adaptive_learning', {})
        self.learned_words = defaultdict(int)
        self.total_samples = 0

    @log_exception
    async def recognize_text(self, image, dpi_scale=None):
        if isinstance(image, str):
            image = cv2.imread(image)
        elif not isinstance(image, np.ndarray):
            raise ValueError("Input must be an image file path or numpy array")

        dpi_scale = dpi_scale or self.default_dpi_scale
        if self.config.get('auto_dpi_detection', False):
            dpi_scale = await self.auto_detect_dpi(image)

        # 根据DPI缩放图像
        if dpi_scale != 1.0:
            image = cv2.resize(image, None, fx=dpi_scale, fy=dpi_scale, interpolation=cv2.INTER_CUBIC)

        # 图像预处理
        preprocessed_image = await self.preprocess_image(image)

        result = self.ocr.ocr(preprocessed_image, cls=True)
        recognized_text = [self.post_process_text(line[1][0]) for line in result if line[1][1] >= self.confidence_threshold]
        
        if self.adaptive_learning.get('enabled', False):
            self.update_learned_words(recognized_text)

        return ' '.join(recognized_text)

    async def auto_detect_dpi(self, image):
        best_scale = self.default_dpi_scale
        best_confidence = 0
        min_scale = self.config.get('min_dpi_scale', 0.5)
        max_scale = self.config.get('max_dpi_scale', 2.0)
        step = self.config.get('dpi_step', 0.1)

        for scale in np.arange(min_scale, max_scale + step, step):
            scaled_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            result = self.ocr.ocr(scaled_image, cls=True)
            confidence = sum(line[1][1] for line in result) / len(result) if result else 0
            if confidence > best_confidence:
                best_confidence = confidence
                best_scale = scale

        return best_scale

    def post_process_text(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        
        return ' '.join(self.correct_word(word) for word in words)

    def correct_word(self, word):
        if word.lower() in self.context_words or self.spell_checker.check(word) or self.is_learned_word(word):
            return word
        suggestions = self.spell_checker.suggest(word)
        if suggestions:
            corrected_word = suggestions[0]
            self.logger.debug(f"Corrected '{word}' to '{corrected_word}'")
            return corrected_word
        return word

    def is_learned_word(self, word):
        return self.learned_words[word.lower()] >= self.adaptive_learning.get('learning_threshold', 5)

    def update_learned_words(self, text):
        words = text.split()
        for word in words:
            self.learned_words[word.lower()] += 1
        self.total_samples += 1
        if self.total_samples > self.adaptive_learning.get('max_samples', 1000):
            self.prune_learned_words()

    def prune_learned_words(self):
        threshold = self.total_samples * self.adaptive_learning.get('learning_rate', 0.01)
        self.learned_words = {word: count for word, count in self.learned_words.items() if count >= threshold}
        self.total_samples = sum(self.learned_words.values())

    async def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        denoise_strength = self.config.get('image_preprocessing', {}).get('denoise_strength', 10)
        denoised = cv2.fastNlMeansDenoising(thresh, None, denoise_strength, 7, 21)
        
        sharpen_strength = self.config.get('image_preprocessing', {}).get('sharpen_strength', 1.5)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * sharpen_strength
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened

    @log_exception
    async def recognize_text_multilingual(self, image, languages=None, dpi_scale=None):
        if not self.config.get('multilingual', {}).get('enabled', False):
            return await self.recognize_text(image, dpi_scale)

        languages = languages or self.config.get('multilingual', {}).get('languages', ['en', 'ch', 'ja'])
        results = {}
        for lang in languages:
            ocr = PaddleOCR(use_angle_cls=True, lang=lang)
            preprocessed_image = await self.preprocess_image(image)
            result = ocr.ocr(preprocessed_image, cls=True)
            recognized_text = [self.post_process_text(line[1][0]) for line in result if line[1][1] >= self.confidence_threshold]
            results[lang] = ' '.join(recognized_text)
        return results

    async def extract_structured_data(self, image):
        if not self.config.get('structured_data_extraction', {}).get('enabled', False):
            return {"text": await self.recognize_text(image)}

        text = await self.recognize_text(image)
        structured_data = {"text": text}

        patterns = self.config.get('structured_data_extraction', {}).get('patterns', [])
        for pattern in patterns:
            matches = re.findall(pattern['regex'], text)
            if matches:
                structured_data[pattern['name']] = matches

        return structured_data

    # ... (other methods remain the same)

ocr_utils = OCRUtils()
