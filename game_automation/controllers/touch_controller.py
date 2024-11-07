import asyncio
import aiohttp

class TouchController:
    def __init__(self, adb_device):
        self.adb_device = adb_device
        self.base_url = f"http://localhost:5037/shell/{self.adb_device}"

    async def execute_adb_command(self, command):
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, data=command) as response:
                return await response.text()

    async def tap(self, x, y):
        command = f"input tap {x} {y}"
        await self.execute_adb_command(command)

    async def swipe(self, start_x, start_y, end_x, end_y, duration):
        command = f"input swipe {start_x} {start_y} {end_x} {end_y} {duration}"
        await self.execute_adb_command(command)

    async def long_press(self, x, y, duration):
        await self.swipe(x, y, x, y, duration)

    async def multi_touch(self, touch_points):
        for point in touch_points:
            await self.tap(point['x'], point['y'])

    async def input_text(self, text):
        command = f"input text {text}"
        await self.execute_adb_command(command)

    async def press_key(self, key_code):
        command = f"input keyevent {key_code}"
        await self.execute_adb_command(command)

    async def zoom(self, start_x1, start_y1, start_x2, start_y2, end_x1, end_y1, end_x2, end_y2, duration):
        command = (f"input swipe {start_x1} {start_y1} {end_x1} {end_y1} {duration} & "
                   f"input swipe {start_x2} {start_y2} {end_x2} {end_y2} {duration}")
        await self.execute_adb_command(command)

if __name__ == "__main__":
    async def test_touch_controller():
        controller = TouchController('test_device')
        await controller.tap(100, 200)
        await controller.swipe(100, 200, 300, 400, 500)
        await controller.long_press(150, 250, 1000)
        await controller.multi_touch([{'x': 100, 'y': 100}, {'x': 200, 'y': 200}])
        await controller.input_text("Hello, World!")
        await controller.press_key(4)  # 4 is the key code for the "Back" button
        await controller.zoom(100, 100, 200, 200, 150, 150, 250, 250, 500)

    asyncio.run(test_touch_controller())