# Game Automation Control Project

[中文](README_zh.md) | [日本語](README_ja.md)

This project implements an advanced game automation control system based on ADB, including OCR prediction, game control modules, AI decision making, and various utilities.

## Features

- Multi-language support (English, Chinese, Japanese)
- Cross-platform compatibility (Windows, macOS, Linux)
- Advanced battle strategies with combo system
- Asynchronous programming for improved performance
- Enhanced error handling and logging
- Improved security measures
- AI-driven decision making using reinforcement learning
- Advanced image recognition with ONNX models
- Comprehensive game analysis and visualization
- Rogue-like game mode with blessing system
- Scene understanding and context-aware decision making
- Dynamic resource allocation for optimized performance
- Plugin system for extensibility

## Project Structure

```
├── game_automation (Game automation module)
│   ├── actions (Game actions)
│   ├── ai (AI decision making)
│   ├── analysis (Game analysis)
│   ├── blessing (Blessing system)
│   ├── controllers (Game controllers)
│   ├── device (Device management)
│   ├── gui (Graphical User Interface)
│   ├── i18n (Internationalization)
│   ├── multimodal (Multimodal analysis)
│   ├── nlp (Natural Language Processing)
│   ├── ocr_prediction (OCR utilities)
│   ├── optimization (Performance optimization)
│   ├── plugins (Plugin system)
│   ├── reasoning (Inference engine)
│   ├── rogue (Rogue-like game mode)
│   ├── scene_understanding (Scene analysis)
│   ├── security (Security measures)
│   ├── testing (Advanced testing tools)
│   ├── visualization (Data visualization)
│   ├── web (Web interface)
│   ├── image_recognition.py
│   └── game_engine.py
├── config (Configuration files)
├── utils (Utility modules)
├── tests (Unit tests)
├── frontend (Web application frontend)
├── main (Main program entry points)
└── README.md (Project documentation)
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/game-automation-suite.git
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install frontend dependencies:
   ```
   cd frontend
   npm install
   ```

## Usage

1. Start the backend server:
   ```
   python main/full_feature_launcher.py
   ```

2. Start the frontend development server:
   ```
   cd frontend
   npm run serve
   ```

3. Access the web interface at `http://localhost:8080`

## Configuration

Edit the configuration files in the `config` directory to customize the automation behavior and game settings. The project supports hot-reloading of configuration files.

Key configuration files:
- `config.yaml`: Main configuration file
- `game_settings.yaml`: Game-specific settings
- `resource_paths.yaml`: Resource file locations
- `deploy.template.yaml`: Deployment configuration template

## API Documentation

For detailed API documentation, please refer to the `docs/api.md` file.

## Running Tests

Execute the test suite:
```
python -m pytest tests
```

For test coverage report:
```
python -m pytest tests --cov=game_automation --cov-report=html
```

## Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Recent Updates

- Migrated OCR system to ONNX for better performance
- Implemented plugin system for extensibility
- Enhanced scene understanding with context-aware decision making
- Improved performance with dynamic resource allocation
- Enhanced security measures with encryption
- Improved internationalization support
- Optimized dependency management
- Added comprehensive test coverage reporting

## Project Entry Points

The project provides multiple entry points for different use cases:

1. **GUI Interface** (Recommended for most users)
   ```
   python main.py
   ```

2. **Command Line Interface** (For advanced users and automation)
   ```
   python -m utils.cli
   ```

3. **Game Engine Only** (For headless operation)
   ```
   python -m game_automation.start_game_engine
   ```

Choose the appropriate entry point based on your needs. The GUI interface provides the most comprehensive control and visualization features.
