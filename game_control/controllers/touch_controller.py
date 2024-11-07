import subprocess

class TouchController:
    def __init__(self, adb_device):
        self.adb_device = adb_device

    def tap(self, x, y):
        subprocess.run(["adb", "-s", self.adb_device, "shell", "input", "tap", str(x), str(y)])

    def swipe(self, start_x, start_y, end_x, end_y, duration):
        subprocess.run(["adb", "-s", self.adb_device, "shell", "input", "swipe", 
                        str(start_x), str(start_y), str(end_x), str(end_y), str(duration)])

    def long_press(self, x, y, duration):
        self.swipe(x, y, x, y, duration)

    def multi_touch(self, touch_points):
        for point in touch_points:
            self.tap(point['x'], point['y'])

    def input_text(self, text):
        subprocess.run(["adb", "-s", self.adb_device, "shell", "input", "text", text])

    def press_key(self, key_code):
        subprocess.run(["adb", "-s", self.adb_device, "shell", "input", "keyevent", str(key_code)])

    def zoom(self, start_x1, start_y1, start_x2, start_y2, end_x1, end_y1, end_x2, end_y2, duration):
        subprocess.run(["adb", "-s", self.adb_device, "shell", "input", "swipe", 
                        str(start_x1), str(start_y1), str(end_x1), str(end_y1), str(duration),
                        "&", "input", "swipe", 
                        str(start_x2), str(start_y2), str(end_x2), str(end_y2), str(duration)])

if __name__ == "__main__":
    # 测试 TouchController 类
    controller = TouchController('test_device')
    controller.tap(100, 200)
    controller.swipe(100, 200, 300, 400, 500)
    controller.long_press(150, 250, 1000)
    controller.multi_touch([{'x': 100, 'y': 100}, {'x': 200, 'y': 200}])
    controller.input_text("Hello, World!")
    controller.press_key(4)  # 4 is the key code for the "Back" button
    controller.zoom(100, 100, 200, 200, 150, 150, 250, 250, 500)