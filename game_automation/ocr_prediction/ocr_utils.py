from utils.error_handler import log_exception
import onnxruntime as ort

# 加载ONNX模型
onnx_session = ort.InferenceSession("path/to/model.onnx")

@log_exception
async def recognize_text(image, dpi_scale=None):
    """识别文本"""
    # 处理图像并进行推理
    # 这里需要添加图像预处理和推理的代码
    return recognized_text

@log_exception
async def recognize_text_multilingual(image, languages=None, dpi_scale=None):
    """多语言文本识别"""
    # 处理图像并进行推理
    # 这里需要添加图像预处理和推理的代码
    return recognized_text
