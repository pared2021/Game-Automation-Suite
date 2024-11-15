# Dependency Management Guide

## 1. Current Dependencies Structure

The project's dependencies are now organized into modular groups for better maintainability and optional installation:

### Core Dependencies (requirements-core.txt)
Required for basic functionality:
- aiohttp: Async HTTP client/server
- aiosqlite: Async SQLite interface
- flask: Web framework
- Pillow: Image processing
- pyyaml: YAML file handling
- watchdog: File system monitoring
- numpy: Numerical computing
- pandas: Data analysis
- onnxruntime: ONNX model inference
- opencv-python-headless: Computer vision
- psutil, sentry-sdk, schedule: System monitoring and scheduling

### AI/ML Dependencies (requirements-ai.txt)
Optional, for advanced automation features:
- torch: Deep learning framework
- torchvision: Computer vision utilities
- scikit-learn: Machine learning utilities
- stable-baselines3: Reinforcement learning
- optuna: Hyperparameter optimization
- matplotlib, seaborn: AI training visualization

### NLP Dependencies (requirements-nlp.txt)
Optional, for text analysis features:
- spacy: NLP toolkit

### Development Tools (requirements-dev.txt)
Optional, for development and debugging:
- pytest, pytest-asyncio: Testing
- black, flake8, mypy: Code quality
- dash, plotly, networkx: Development visualization

## 2. Installation Guidelines

### 2.1 Basic Installation (Core Features)
```bash
pip install -r requirements.txt
```

### 2.2 Full Installation (All Features)
```bash
pip install -r requirements.txt -r requirements-ai.txt -r requirements-nlp.txt
```

### 2.3 Development Installation
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## 3. Dependency Management Guidelines

### 3.1 Version Constraints
- All dependencies use >= version constraints
- Version numbers specified to known-working versions
- Major version updates require thorough testing

### 3.2 Framework Standards
- PyTorch is the standard deep learning framework
- ONNX used for model deployment
- OpenCV for computer vision tasks
- Spacy for NLP tasks

### 3.3 Adding New Dependencies
When adding new dependencies:
1. Determine appropriate dependency group
2. Check for overlap with existing packages
3. Prefer well-maintained packages
4. Test compatibility
5. Update documentation

## 4. Recent Optimizations

### 4.1 Completed Optimizations
- Reorganized dependencies into modular groups
- Removed unused dependencies:
  * transformers (imported but unused)
  * nltk (unused)
  * tensorflow (standardized on PyTorch)
  * paddleocr (migrated to ONNX)
  * librosa (unused)
  * keras (standardized on PyTorch)
  * gym (using stable-baselines3)
- Implemented modular installation system

### 4.2 Benefits
- Reduced base installation size
- Clearer dependency structure
- Easier maintenance
- Optional feature installation
- Better development experience

## 5. Future Considerations

### 5.1 Planned Improvements
- Regular dependency audits
- Automated compatibility testing
- Container-based dependency isolation
- Version compatibility matrix
- Dependency usage monitoring

### 5.2 Maintenance Tasks
- Regular security updates
- Periodic dependency reviews
- Usage analysis
- Version constraint updates
- Breaking change documentation

## 6. Troubleshooting

### 6.1 Common Issues
- Version conflicts: Check compatibility matrix
- Installation failures: Verify system prerequisites
- Memory issues: Consider partial installation

### 6.2 Support Process
1. Check installation logs
2. Verify dependency group requirements
3. Review compatibility documentation
4. Contact project maintainers if needed
