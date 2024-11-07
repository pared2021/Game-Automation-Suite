import json
import os
import yaml
from typing import Dict, Any, Optional
from utils.error_handler import log_exception
from utils.logger import detailed_logger

class I18nManager:
    """
    统一的国际化管理类，支持JSON和YAML格式的翻译文件
    """
    def __init__(self, default_language: str = 'en-US'):
        """
        初始化国际化管理器
        
        Args:
            default_language: 默认语言代码
        """
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, Any]] = {}
        self.logger = detailed_logger
        self.load_translations()

    @log_exception
    def load_translations(self) -> None:
        """加载所有可用的翻译文件"""
        i18n_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'i18n')
        if not os.path.exists(i18n_dir):
            self.logger.warning(f"Translation directory not found: {i18n_dir}")
            return

        for filename in os.listdir(i18n_dir):
            if filename.endswith(('.json', '.yaml', '.yml')):
                language = filename.split('.')[0]
                file_path = os.path.join(i18n_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if filename.endswith('.json'):
                            self.translations[language] = json.load(f)
                        else:  # yaml or yml
                            self.translations[language] = yaml.safe_load(f)
                    self.logger.info(f"Loaded translations for language: {language}")
                except json.JSONDecodeError:
                    self.logger.error(f"Error decoding JSON in file: {file_path}")
                except yaml.YAMLError:
                    self.logger.error(f"Error decoding YAML in file: {file_path}")
                except IOError as e:
                    self.logger.error(f"Error reading file {file_path}: {str(e)}")

    @log_exception
    def set_language(self, language: str) -> None:
        """
        设置当前使用的语言
        
        Args:
            language: 语言代码
        """
        if language in self.translations:
            self.current_language = language
            self.logger.info(f"Language set to: {language}")
        else:
            self.logger.warning(f"Unsupported language: {language}. Falling back to {self.default_language}")
            self.current_language = self.default_language

    @log_exception
    def get(self, key: str, **kwargs: Any) -> str:
        """
        获取指定键的翻译文本
        
        Args:
            key: 翻译键，支持点号分隔的嵌套键
            **kwargs: 用于格式化翻译文本的参数

        Returns:
            翻译后的文本，如果未找到则返回键名
        """
        try:
            translation = self.translations.get(self.current_language, {})
            for part in key.split('.'):
                translation = translation.get(part, {})
            
            if isinstance(translation, str):
                return translation.format(**kwargs)
            return key
        except KeyError:
            self.logger.warning(f"Translation key not found: {key}")
            return key
        except Exception as e:
            self.logger.error(f"Error getting translation for key {key}: {str(e)}")
            return key

    @log_exception
    def add_language(self, language: str, translations: Dict[str, Any]) -> None:
        """
        添加或更新语言翻译
        
        Args:
            language: 语言代码
            translations: 翻译字典
        """
        if language not in self.translations:
            self.translations[language] = translations
            self.logger.info(f"Added new language: {language}")
        else:
            self.translations[language].update(translations)
            self.logger.info(f"Updated translations for language: {language}")

    def t(self, key: str, **kwargs: Any) -> str:
        """
        获取翻译的简便方法
        
        Args:
            key: 翻译键
            **kwargs: 格式化参数

        Returns:
            翻译后的文本
        """
        return self.get(key, **kwargs)

    @property
    def available_languages(self) -> list:
        """获取所有可用的语言列表"""
        return list(self.translations.keys())

    @property
    def current_lang(self) -> str:
        """获取当前使用的语言"""
        return self.current_language

# 创建全局实例
i18n = I18nManager()
