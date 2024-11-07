import json
import os
from typing import Dict, Any
from utils.error_handler import log_exception
from utils.logger import detailed_logger

class I18nManager:
    def __init__(self, default_language: str = 'en-US'):
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, Any]] = {}
        self.logger = detailed_logger
        self.load_translations()

    @log_exception
    def load_translations(self) -> None:
        i18n_dir = os.path.join(os.path.dirname(__file__), '..', 'config', 'i18n')
        for filename in os.listdir(i18n_dir):
            if filename.endswith('.json'):
                language = filename.split('.')[0]
                file_path = os.path.join(i18n_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.translations[language] = json.load(f)
                except json.JSONDecodeError:
                    self.logger.error(f"Error decoding JSON in file: {file_path}")
                except IOError:
                    self.logger.error(f"Error reading file: {file_path}")

    @log_exception
    def set_language(self, language: str) -> None:
        if language in self.translations:
            self.current_language = language
        else:
            self.logger.warning(f"Unsupported language: {language}. Falling back to default.")
            self.current_language = self.default_language

    @log_exception
    def get(self, key: str, **kwargs: Any) -> str:
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

i18n_manager = I18nManager()
