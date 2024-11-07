import json
import os
import yaml
from utils.error_handler import log_exception
from utils.logger import detailed_logger

class I18nManager:
    def __init__(self, default_language='en'):
        self.default_language = default_language
        self.current_language = default_language
        self.translations = {}
        self.logger = detailed_logger
        self.load_translations()

    @log_exception
    def load_translations(self):
        # 加载翻译文件
        i18n_dir = os.path.join(os.path.dirname(__file__), '..', 'config', 'i18n')
        for filename in os.listdir(i18n_dir):
            if filename.endswith('.json') or filename.endswith('.yaml'):
                language = filename.split('.')[0]
                file_path = os.path.join(i18n_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    if filename.endswith('.json'):
                        self.translations[language] = json.load(f)
                    elif filename.endswith('.yaml'):
                        self.translations[language] = yaml.safe_load(f)

    @log_exception
    def set_language(self, language):
        # 设置当前语言
        if language in self.translations:
            self.current_language = language
        else:
            self.logger.warning(f"Unsupported language: {language}. Falling back to default.")
            self.current_language = self.default_language

    @log_exception
    def get(self, key, **kwargs):
        # 获取翻译
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
    def add_language(self, language, translations):
        # 添加新语言
        if language not in self.translations:
            self.translations[language] = translations
        else:
            self.translations[language].update(translations)

i18n_manager = I18nManager()
