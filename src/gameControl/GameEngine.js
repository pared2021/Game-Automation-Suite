const adb = require('adbkit');
const fs = require('fs').promises;
const path = require('path');
const TouchController = require('./controllers/TouchController');
const AutoBattleStrategy = require('./controllers/AutoBattleStrategy');
const OCRUtils = require('./ocrPrediction/OCRUtils');
const ImageRecognition = require('./ImageRecognition');

class GameEngine {
  constructor(strategyFile, adbDevice = null) {
    this.adbDevice = adbDevice;
    this.touchController = null;
    this.battleStrategy = new AutoBattleStrategy(strategyFile);
    this.ocrUtils = new OCRUtils();
    this.imageRecognition = new ImageRecognition();
    this.stopAutomation = false;
    this.screenWidth = 0;
    this.screenHeight = 0;
    this.emulatorSettings = null;
    this.client = adb.createClient();
  }

  async initialize() {
    await this.loadEmulatorSettings();
    await this.detectEmulators();
    await this.selectEmulator(this.adbDevice);
    await this.updateScreenSize();
  }

  async loadEmulatorSettings() {
    const data = await fs.readFile(path.join(__dirname, '../../config/emulator_settings.json'), 'utf8');
    this.emulatorSettings = JSON.parse(data);
  }

  async detectEmulators() {
    const devices = await this.client.listDevices();
    this.emulatorSettings.emulators = devices.map(device => ({
      name: `Emulator_${device.id}`,
      serial: device.id,
      status: device.type
    }));
  }

  async selectEmulator(adbDevice) {
    if (adbDevice) {
      const emulator = this.emulatorSettings.emulators.find(e => e.serial === adbDevice);
      if (emulator) {
        this.adbDevice = adbDevice;
      } else {
        throw new Error(`Specified ADB device ${adbDevice} not found`);
      }
    } else {
      if (this.emulatorSettings.emulators.length > 0) {
        this.adbDevice = this.emulatorSettings.emulators[0].serial;
      } else {
        throw new Error("No ADB devices detected");
      }
    }
    
    this.touchController = new TouchController(this.adbDevice);
  }

  async updateScreenSize() {
    const output = await this.client.shell(this.adbDevice, 'wm size');
    const size = output.toString().trim().split(':')[1].trim().split('x');
    [this.screenWidth, this.screenHeight] = size.map(Number);
  }

  async captureScreen() {
    const screenshotBuffer = await this.client.screencap(this.adbDevice);
    return screenshotBuffer;
  }

  async findGameObject(templatePath) {
    const screen = await this.captureScreen();
    return this.imageRecognition.templateMatching(screen, templatePath);
  }

  async waitForGameObject(templatePath, timeout = 30000) {
    const startTime = Date.now();
    while (Date.now() - startTime < timeout) {
      const result = await this.findGameObject(templatePath);
      if (result.length > 0) {
        return true;
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    return false;
  }

  async performActionSequence(actions) {
    for (const action of actions) {
      if (action.type === 'tap') {
        await this.touchController.tap(action.x, action.y);
      } else if (action.type === 'wait') {
        await new Promise(resolve => setTimeout(resolve, action.duration));
      } else if (action.type === 'find_and_tap') {
        const objects = await this.findGameObject(action.template);
        if (objects.length > 0) {
          const [x, y, w, h] = objects[0];
          await this.touchController.tap(x + w/2, y + h/2);
        }
      }
    }
  }

  async detectGameState() {
    // Implement game state detection logic
    // This is a placeholder and should be implemented based on your game's specific requirements
    return 'unknown';
  }

  async getResourceValue(resource) {
    // Implement resource value retrieval logic
    // This is a placeholder and should be implemented based on your game's specific requirements
    return 0;
  }

  getEmulatorInfo() {
    return this.emulatorSettings.emulators.find(e => e.serial === this.adbDevice);
  }

  async startEmulator() {
    console.log("Starting emulator...");
    // Implement the logic to start the emulator
    // This might involve running a shell command or using an emulator-specific API
  }

  async runGameLoop() {
    while (!this.stopAutomation) {
      try {
        const screen = await this.captureScreen();
        const text = await this.ocrUtils.recognizeText(screen);
        
        if (text.includes("战斗")) {
          await this.battleStrategy.executeStrategy("battle", this);
        } else if (text.includes("Boss")) {
          await this.battleStrategy.executeStrategy("boss_battle", this);
        } else {
          await this.battleStrategy.executeStrategy("default", this);
        }
        
        await new Promise(resolve => setTimeout(resolve, 1000));
      } catch (error) {
        console.error("Error in game loop:", error);
      }
    }
  }
}

module.exports = GameEngine;