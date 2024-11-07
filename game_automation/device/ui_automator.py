import uiautomator2 as u2

class UIAutomator:
    def __init__(self, device_id):
        self.device = u2.connect(device_id)

    def click(self, x, y):
        self.device.click(x, y)

    def swipe(self, fx, fy, tx, ty, duration=0.1):
        self.device.swipe(fx, fy, tx, ty, duration)

    def text(self, text):
        self.device(text=text).click()

    def wait(self, timeout=10.0):
        return self.device.wait_activity(timeout)

    def get_device_info(self):
        return self.device.info

    def screenshot(self, filename):
        self.device.screenshot(filename)

    def app_start(self, package_name):
        self.device.app_start(package_name)

    def app_stop(self, package_name):
        self.device.app_stop(package_name)