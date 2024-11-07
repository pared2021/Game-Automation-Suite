import json
import os
from utils.logger import setup_logger
from utils.config_manager import config_manager

class Internationalization:
    def __init__(self):
        self.logger = setup_logger('internationalization')
        self.config = config_manager.get('i18n', {})
        self.current_language = self.config.get('default_language', 'en')
        self.translations = {}
        self.load_translations()

    def load_translations(self):
        lang_dir = 'game_automation/i18n/lang'
        for filename in os.listdir(lang_dir):
            if filename.endswith('.json'):
                lang_code = filename[:-5]
                with open(os.path.join(lang_dir, filename), 'r', encoding='utf-8') as f:
                    self.translations[lang_code] = json.load(f)
        self.logger.info(f"Loaded translations for {len(self.translations)} languages")

    def set_language(self, lang_code):
        if lang_code in self.translations:
            self.current_language = lang_code
            self.logger.info(f"Language set to {lang_code}")
        else:
            self.logger.warning(f"Unsupported language: {lang_code}")

    def get_text(self, key):
        return self.translations.get(self.current_language, {}).get(key, key)

i18n = Internationalization()