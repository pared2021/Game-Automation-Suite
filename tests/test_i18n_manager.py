import unittest
from unittest.mock import patch, mock_open
from game_automation.utils.i18n_manager import I18nManager

class TestI18nManager(unittest.TestCase):
    def setUp(self):
        self.i18n = I18nManager()

    @patch('os.listdir')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_translations(self, mock_file, mock_listdir):
        mock_listdir.return_value = ['en-US.json', 'zh-CN.json']
        mock_file.return_value.__enter__.return_value.read.side_effect = [
            '{"test": "Test"}',
            '{"test": "测试"}'
        ]
        
        self.i18n.load_translations()
        
        self.assertEqual(self.i18n.translations['en-US'], {'test': 'Test'})
        self.assertEqual(self.i18n.translations['zh-CN'], {'test': '测试'})

    def test_set_language(self):
        self.i18n.translations = {'en-US': {}, 'zh-CN': {}}
        self.i18n.set_language('zh-CN')
        self.assertEqual(self.i18n.current_language, 'zh-CN')

        with self.assertRaises(ValueError):
            self.i18n.set_language('fr-FR')

    def test_get(self):
        self.i18n.translations = {
            'en-US': {'test': {'nested': 'Nested test {param}'}},
            'zh-CN': {'test': {'nested': '嵌套测试 {param}'}}
        }
        self.i18n.current_language = 'en-US'
        
        self.assertEqual(self.i18n.get('test.nested', param='value'), 'Nested test value')
        self.assertEqual(self.i18n.get('nonexistent.key'), 'nonexistent.key')

        self.i18n.current_language = 'zh-CN'
        self.assertEqual(self.i18n.get('test.nested', param='值'), '嵌套测试 值')

    def test_add_language(self):
        self.i18n.add_language('fr-FR', {'test': 'Test en français'})
        self.assertEqual(self.i18n.translations['fr-FR'], {'test': 'Test en français'})

        self.i18n.add_language('fr-FR', {'new': 'Nouveau test'})
        self.assertEqual(self.i18n.translations['fr-FR'], {'test': 'Test en français', 'new': 'Nouveau test'})

if __name__ == '__main__':
    unittest.main()