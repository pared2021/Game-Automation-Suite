from PIL import Image

def check_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # 验证图像文件
        print("Image is valid.")
    except Exception as e:
        print(f"Image is not valid: {e}")

if __name__ == "__main__":
    check_image("test_image.jpg")
