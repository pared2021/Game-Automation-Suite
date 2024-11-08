# OCR Prediction Module

这个模块提供了多引擎的OCR文本识别功能，支持PaddleOCR和ONNX两种实现。

## 安装

1. 安装依赖:
```bash
pip install -r requirements.txt
```

2. 下载模型文件:
- 对于PaddleOCR引擎，模型会在首次使用时自动下载
- 对于ONNX引擎，需要手动下载模型文件并放置在 `models/onnx_ocr` 目录下:
  - det.onnx: 文本检测模型
  - rec.onnx: 文本识别模型
  - cls.onnx: 文本方向分类模型
  - ppocr_keys_v1.txt: 字符字典文件

## 使用示例

```python
from ocr_prediction.ocr_utils import OCRUtils, OCREngine

# 使用PaddleOCR引擎(默认)
ocr = OCRUtils()

# 或使用ONNX引擎
ocr = OCRUtils(engine=OCREngine.ONNX)

# 识别图片中的文本
text = ocr.recognize_text("image.jpg")
print(f"识别结果: {text}")

# 识别指定区域的文本
region = (100, 100, 200, 50)  # x, y, width, height
text = ocr.extract_text_from_region("image.jpg", region)
print(f"区域文本: {text}")

# 获取可用的OCR引擎
engines = OCRUtils.get_available_engines()
print(f"可用引擎: {engines}")
```

## 引擎特点

1. PaddleOCR引擎:
- 使用飞桨深度学习框架
- 支持中英文混合识别
- 准确率较高
- 首次使用时需要下载模型

2. ONNX引擎:
- 使用ONNX Runtime推理框架
- 部署更加灵活
- 跨平台兼容性好
- 需要手动下载模型文件

## 注意事项

1. 两种引擎的识别结果可能略有差异
2. ONNX引擎需要预先下载模型文件
3. 建议根据具体使用场景选择合适的引擎
