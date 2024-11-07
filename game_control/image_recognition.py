import cv2
import numpy as np
from functools import lru_cache

class ImageRecognition:
    @staticmethod
    @lru_cache(maxsize=32)
    def load_template(template_path):
        return cv2.imread(template_path, 0)

    @staticmethod
    def template_matching(screen, template_path, threshold=0.8):
        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        template = ImageRecognition.load_template(template_path)
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        matches = []
        for pt in zip(*loc[::-1]):
            matches.append((pt[0], pt[1], w, h))

        return matches

    @staticmethod
    def find_color(screen, color, tolerance=10):
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        lower = np.array([max(0, color[0] - tolerance), max(0, color[1] - tolerance), max(0, color[2] - tolerance)])
        upper = np.array([min(255, color[0] + tolerance), min(255, color[1] + tolerance), min(255, color[2] + tolerance)])

        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return [cv2.boundingRect(c) for c in contours]

    @staticmethod
    def detect_edges(screen):
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges

    @staticmethod
    def find_contours(screen):
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def match_feature_points(screen1, screen2):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(screen1, None)
        kp2, des2 = orb.detectAndCompute(screen2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        return matches[:10]  # Return top 10 matches

if __name__ == "__main__":
    # 测试 ImageRecognition 类
    test_image = cv2.imread('test_image.png')
    matches = ImageRecognition.template_matching(test_image, 'template.png')
    print(f"Template matches: {matches}")
    color_regions = ImageRecognition.find_color(test_image, [255, 0, 0])  # 查找红色区域
    print(f"Color regions: {color_regions}")
    edges = ImageRecognition.detect_edges(test_image)
    cv2.imwrite('edges.png', edges)
    contours = ImageRecognition.find_contours(test_image)
    print(f"Number of contours: {len(contours)}")
    if cv2.imread('test_image2.png') is not None:
        feature_matches = ImageRecognition.match_feature_points(test_image, cv2.imread('test_image2.png'))
        print(f"Number of feature matches: {len(feature_matches)}")