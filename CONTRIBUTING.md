# Contributing to Game Automation Suite

感谢您对Game Automation Suite的贡献兴趣！本文档将指导您如何参与项目开发。

## 开发环境设置

1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/game-automation-suite.git
   cd game-automation-suite
   ```

2. 安装依赖：
   ```bash
   make install
   ```
   这将安装所有必要的依赖并设置pre-commit钩子。

## 开发工作流程

1. 创建新分支：
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/your-fix-name
   ```

2. 开发时运行测试：
   ```bash
   make test        # 运行测试
   make coverage    # 生成覆盖率报告
   ```

3. 代码质量检查：
   ```bash
   make lint        # 运行代码质量检查
   make format      # 格式化代码
   ```

4. 提交前检查：
   ```bash
   make check       # 运行所有检查（lint、test、security）
   ```

## 代码规范

1. Python代码规范：
   - 遵循PEP 8规范
   - 使用Black进行代码格式化
   - 使用类型注解
   - 保持函数简洁，单一职责
   - 编写清晰的文档字符串

2. 测试规范：
   - 所有新功能必须包含测试
   - 测试覆盖率要求不低于80%
   - 使用pytest进行测试
   - 使用fixture减少代码重复

3. 提交规范：
   - 使用清晰的提交消息
   - 每个提交专注于单一改动
   - 提交消息格式：
     ```
     type(scope): description
     
     [optional body]
     [optional footer]
     ```
   - 类型包括：
     - feat: 新功能
     - fix: 错误修复
     - docs: 文档更改
     - style: 代码格式调整
     - refactor: 代码重构
     - test: 测试相关
     - chore: 构建过程或辅助工具的变动

## 项目结构

```
game_automation/
├── actions/         # 游戏动作模块
├── ai/             # AI决策模块
├── core/           # 核心功能
├── device/         # 设备管理
├── i18n/           # 国际化支持
├── utils/          # 工具函数
└── web/            # Web界面
```

## 开发指南

1. 国际化支持：
   - 所有用户界面文本必须使用i18n系统
   - 支持的语言：英语、中文、日语
   - 翻译文件位于 `game_automation/i18n/locales/`

2. 错误处理：
   - 使用适当的异常类型
   - 提供清晰的错误消息
   - 记录关键错误日志

3. 日志记录：
   - 使用适当的日志级别
   - 包含必要的上下文信息
   - 避免敏感信息泄露

4. 性能考虑：
   - 注意内存使用
   - 优化循环和算法
   - 使用性能分析工具

## 发布流程

1. 版本号规范：
   - 遵循语义化版本（Semantic Versioning）
   - 格式：MAJOR.MINOR.PATCH

2. 发布检查清单：
   - 所有测试通过
   - 文档更新
   - 更新日志完善
   - 依赖列表更新

## 问题报告

报告问题时请包含：
- 问题的详细描述
- 复现步骤
- 期望行为
- 实际行为
- 环境信息（操作系统、Python版本等）
- 相关日志或截图

## 帮助和支持

- 查看项目文档
- 提交Issue
- 加入开发者讨论组

## 许可证

贡献的代码将在MIT许可证下发布。
