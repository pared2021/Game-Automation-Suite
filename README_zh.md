# 游戏自动化控制项目

[English](README.md) | [日本語](README_ja.md)

本项目实现了一个基于ADB的高级游戏自动化控制系统，包括OCR预测、游戏控制模块、AI决策制定和各种实用工具。

## 功能特性

- 多语言支持（英语、中文、日语）
- 跨平台兼容（Windows、macOS、Linux）
- 高级战斗策略系统
- 异步编程提升性能
- 增强的错误处理和日志记录
- 改进的安全措施
- 基于强化学习的AI决策系统
- 基于ONNX的高级图像识别
- 全面的游戏分析和可视化
- 集成祝福系统的Rogue模式
- 场景理解和上下文感知决策
- 动态资源分配优化性能
- 插件系统支持扩展功能

## 项目结构

```
├── game_automation (游戏自动化模块)
│   ├── actions (游戏动作)
│   ├── ai (AI决策制定)
│   ├── analysis (游戏分析)
│   ├── blessing (祝福系统)
│   ├── controllers (游戏控制器)
│   ├── device (设备管理)
│   ├── gui (图形用户界面)
│   ├── i18n (国际化)
│   ├── multimodal (多模态分析)
│   ├── nlp (自然语言处理)
│   ├── ocr_prediction (OCR工具)
│   ├── optimization (性能优化)
│   ├── plugins (插件系统)
│   ├── reasoning (推理引擎)
│   ├── rogue (Rogue模式)
│   ├── scene_understanding (场景分析)
│   ├── security (安全措施)
│   ├── testing (测试工具)
│   ├── visualization (数据可视化)
│   ├── web (Web界面)
│   ├── image_recognition.py
│   └── game_engine.py
├── config (配置文件)
├── utils (工具模块)
├── tests (单元测试)
├── frontend (Web应用前端)
├── main (主程序入口)
└── README.md (项目文档)
```

## 安装说明

1. 克隆仓库：
   ```
   git clone https://github.com/yourusername/game-automation-suite.git
   ```

2. 安装Python依赖：
   ```
   pip install -r requirements.txt
   ```

3. 安装前端依赖：
   ```
   cd frontend
   npm install
   ```

## 使用方法

1. 启动后端服务器：
   ```
   python main/full_feature_launcher.py
   ```

2. 启动前端开发服务器：
   ```
   cd frontend
   npm run serve
   ```

3. 在浏览器中访问 `http://localhost:8080` 使用Web界面

## 配置说明

在 `config` 目录中编辑配置文件来自定义自动化行为和游戏设置。本项目支持配置文件热重载。

主要配置文件：
- `config.yaml`: 主配置文件
- `game_settings.yaml`: 游戏专用设置
- `resource_paths.yaml`: 资源文件路径
- `deploy.template.yaml`: 部署配置模板

## API文档

详细的API文档请参考 `docs/api.md` 文件。

## 运行测试

执行测试套件：
```
python -m pytest tests
```

生成测试覆盖率报告：
```
python -m pytest tests --cov=game_automation --cov-report=html
```

## 参与贡献

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交Pull Request

## 开源协议

本项目采用 MIT 许可证。详情请参阅 `LICENSE` 文件。

## 最近更新

- OCR系统迁移至ONNX以提升性能
- 实现插件系统支持功能扩展
- 增强场景理解的上下文感知决策能力
- 通过动态资源分配提升性能
- 加强安全措施和加密功能
- 改进国际化支持
- 优化依赖管理
- 添加全面的测试覆盖率报告
