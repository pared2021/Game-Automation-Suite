import json
import os

class I18n:
    def __init__(self, lang='en-US'):
        self.lang = lang
        self.translations = {}
        self.load_translations()

    def load_translations(self):
        # 加载翻译文件
        file_path = os.path.join('config', 'i18n', f'{self.lang}.json')
        with open(file_path, 'r', encoding='utf-8') as f:
            self.translations = json.load(f)

    def t(self, key):
        # 获取翻译
        keys = key.split('.')
        value = self.translations
        for k in keys:
            if k in value:
                value = value[k]
            else:
                return key
        return value if isinstance(value, str) else key

    def set_language(self, lang):
        # 设置语言
        self.lang = lang
        self.load_translations()

i18n = I18n()  # 创建全局实例

# 使用示例:
# print(i18n.t('general.start'))
# i18n.set_language('zh-CN')
# print(i18n.t('general.start'))