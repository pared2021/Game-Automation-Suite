import os
import json
import asyncio
import subprocess
from typing import List, Dict, Optional
from pathlib import Path

from utils.error_handler import log_exception, GameAutomationError
from utils.logger import detailed_logger

class EmulatorError(GameAutomationError):
    """模拟器相关错误"""
    pass

class EmulatorManager:
    """模拟器管理器，负责模拟器检测、连接和配置管理"""

    # 常见模拟器的默认安装路径
    EMULATOR_PATHS = {
        'nox': [
            r'C:\Program Files\Nox\bin',
            r'C:\Program Files (x86)\Nox\bin'
        ],
        'bluestacks': [
            r'C:\Program Files\BlueStacks_nxt',
            r'C:\Program Files (x86)\BlueStacks_nxt'
        ],
        'memu': [
            r'C:\Program Files\MEmu',
            r'C:\Program Files (x86)\MEmu'
        ]
    }

    def __init__(self):
        """初始化模拟器管理器"""
        self.emulator_type: Optional[str] = None
        self.emulator_path: Optional[str] = None
        self.emulator_config: Dict = {}
        self.connected: bool = False

    @log_exception
    async def detect_emulators(self) -> List[Dict[str, str]]:
        """检测已安装的模拟器
        
        Returns:
            List[Dict[str, str]]: 检测到的模拟器列表，每个字典包含类型和路径
        """
        detected_emulators = []
        
        for emu_type, paths in self.EMULATOR_PATHS.items():
            for path in paths:
                if os.path.exists(path):
                    # 检查特定于模拟器类型的可执行文件
                    exe_name = self._get_emulator_exe_name(emu_type)
                    exe_path = os.path.join(path, exe_name)
                    
                    if os.path.exists(exe_path):
                        detected_emulators.append({
                            'type': emu_type,
                            'path': path,
                            'executable': exe_path
                        })
                        detailed_logger.info(f"检测到{emu_type}模拟器: {path}")

        if not detected_emulators:
            detailed_logger.warning("未检测到已安装的模拟器")
        
        return detected_emulators

    @log_exception
    async def connect(self, emulator_type: str = None, custom_path: str = None) -> bool:
        """连接到指定类型的模拟器
        
        Args:
            emulator_type: 模拟器类型 ('nox', 'bluestacks', 'memu')
            custom_path: 自定义模拟器路径
            
        Returns:
            bool: 连接是否成功
            
        Raises:
            EmulatorError: 模拟器连接失败时抛出
        """
        try:
            if custom_path and os.path.exists(custom_path):
                self.emulator_path = custom_path
                # 尝试从路径推断模拟器类型
                self.emulator_type = self._detect_emulator_type(custom_path)
            elif emulator_type:
                # 使用默认路径
                detected = await self.detect_emulators()
                matching = [e for e in detected if e['type'] == emulator_type]
                if matching:
                    self.emulator_type = emulator_type
                    self.emulator_path = matching[0]['path']
                else:
                    raise EmulatorError(f"未找到类型为 {emulator_type} 的模拟器")
            else:
                # 自动选择第一个检测到的模拟器
                detected = await self.detect_emulators()
                if detected:
                    self.emulator_type = detected[0]['type']
                    self.emulator_path = detected[0]['path']
                else:
                    raise EmulatorError("未检测到任何模拟器")

            # 加载模拟器配置
            await self.load_emulator_settings()
            
            # 获取模拟器ADB端口
            adb_port = await self._get_emulator_adb_port()
            if not adb_port:
                raise EmulatorError("无法获取模拟器ADB端口")

            # 连接到模拟器
            connect_cmd = f"adb connect 127.0.0.1:{adb_port}"
            process = await asyncio.create_subprocess_shell(
                connect_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                self.connected = True
                detailed_logger.info(f"成功连接到{self.emulator_type}模拟器")
                return True
            else:
                raise EmulatorError(f"连接模拟器失败: {stderr.decode()}")

        except Exception as e:
            detailed_logger.error(f"模拟器连接失败: {str(e)}")
            raise EmulatorError(f"模拟器连接失败: {str(e)}")

    @log_exception
    async def load_emulator_settings(self) -> None:
        """加载模拟器设置
        
        Raises:
            EmulatorError: 加载设置失败时抛出
        """
        if not self.emulator_type or not self.emulator_path:
            raise EmulatorError("未指定模拟器类型或路径")

        config_path = self._get_config_path()
        if not config_path:
            detailed_logger.warning("未找到模拟器配置文件")
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.emulator_config = json.load(f)
                detailed_logger.info(f"成功加载{self.emulator_type}模拟器配置")
        except Exception as e:
            detailed_logger.error(f"加载模拟器配置失败: {str(e)}")
            self.emulator_config = {}

    @log_exception
    async def start_emulator(self, instance_name: str = None) -> bool:
        """启动模拟器
        
        Args:
            instance_name: 模拟器实例名称（可选）
            
        Returns:
            bool: 启动是否成功
        """
        if not self.emulator_type or not self.emulator_path:
            raise EmulatorError("未指定模拟器类型或路径")

        try:
            exe_path = os.path.join(self.emulator_path, self._get_emulator_exe_name(self.emulator_type))
            cmd = [exe_path]
            
            if instance_name:
                if self.emulator_type == 'nox':
                    cmd.extend(['-clone:', instance_name])
                elif self.emulator_type == 'bluestacks':
                    cmd.extend(['--instance', instance_name])

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # 等待模拟器启动
            await asyncio.sleep(10)
            
            # 检查进程状态
            if process.returncode is None or process.returncode == 0:
                detailed_logger.info(f"成功启动{self.emulator_type}模拟器")
                return True
            else:
                detailed_logger.error(f"启动模拟器失败，返回码: {process.returncode}")
                return False

        except Exception as e:
            detailed_logger.error(f"启动模拟器失败: {str(e)}")
            return False

    def _get_emulator_exe_name(self, emulator_type: str) -> str:
        """获取模拟器可执行文件名称"""
        exe_names = {
            'nox': 'Nox.exe',
            'bluestacks': 'HD-Player.exe',
            'memu': 'MEmu.exe'
        }
        return exe_names.get(emulator_type, '')

    def _detect_emulator_type(self, path: str) -> Optional[str]:
        """从路径推断模拟器类型"""
        path_lower = path.lower()
        if 'nox' in path_lower:
            return 'nox'
        elif 'bluestacks' in path_lower:
            return 'bluestacks'
        elif 'memu' in path_lower:
            return 'memu'
        return None

    def _get_config_path(self) -> Optional[str]:
        """获取模拟器配置文件路径"""
        if not self.emulator_type or not self.emulator_path:
            return None

        config_paths = {
            'nox': os.path.join(self.emulator_path, 'conf', 'config.ini'),
            'bluestacks': os.path.join(self.emulator_path, 'BlueStacks.conf'),
            'memu': os.path.join(self.emulator_path, 'config.ini')
        }
        
        config_path = config_paths.get(self.emulator_type)
        return config_path if config_path and os.path.exists(config_path) else None

    async def _get_emulator_adb_port(self) -> Optional[int]:
        """获取模拟器ADB端口"""
        default_ports = {
            'nox': 62001,
            'bluestacks': 5555,
            'memu': 21503
        }
        
        # 首先尝试从配置文件获取
        if self.emulator_config:
            # 根据不同模拟器类型解析配置
            if self.emulator_type == 'nox':
                return self.emulator_config.get('adb_port', default_ports['nox'])
            elif self.emulator_type == 'bluestacks':
                return self.emulator_config.get('bst.instance.Pie64.status.adb_port', default_ports['bluestacks'])
            elif self.emulator_type == 'memu':
                return self.emulator_config.get('adb_port', default_ports['memu'])

        # 如果无法从配置获取，使用默认端口
        return default_ports.get(self.emulator_type)

    @property
    def is_connected(self) -> bool:
        """获取模拟器连接状态"""
        return self.connected

    @property
    def current_emulator(self) -> Dict[str, str]:
        """获取当前模拟器信息"""
        return {
            'type': self.emulator_type,
            'path': self.emulator_path,
            'connected': self.connected
        }
