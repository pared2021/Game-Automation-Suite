import pytest
from game_automation.i18n.internationalization import I18nManager

@pytest.fixture
def i18n():
    """Create a fresh I18nManager instance for each test."""
    return I18nManager(default_language="en-US")

def test_i18n_initialization(i18n):
    """Test basic initialization of I18nManager."""
    assert i18n.default_language == "en-US"
    assert i18n.current_language == "en-US"
    assert isinstance(i18n.translations, dict)
    assert len(i18n.supported_languages) == 3
    assert "en-US" in i18n.supported_languages
    assert "zh-CN" in i18n.supported_languages
    assert "ja-JP" in i18n.supported_languages

def test_language_switching(i18n):
    """Test language switching functionality."""
    # Test valid language switch
    assert i18n.set_language("zh-CN") is True
    assert i18n.current_language == "zh-CN"
    
    # Test invalid language switch
    assert i18n.set_language("invalid-lang") is False
    assert i18n.current_language == "zh-CN"  # Should remain unchanged

def test_text_translation(i18n):
    """Test text translation functionality."""
    # Test English text
    i18n.set_language("en-US")
    assert i18n.get_text("common.ok") == "OK"
    assert i18n.get_text("common.cancel") == "Cancel"

    # Test Chinese text
    i18n.set_language("zh-CN")
    assert i18n.get_text("common.ok") == "确定"
    assert i18n.get_text("common.cancel") == "取消"

    # Test Japanese text
    i18n.set_language("ja-JP")
    assert i18n.get_text("common.ok") == "OK"
    assert i18n.get_text("common.cancel") == "キャンセル"

def test_nested_translation_keys(i18n):
    """Test accessing nested translation keys."""
    i18n.set_language("en-US")
    assert i18n.get_text("menu.settings") == "Settings"
    assert i18n.get_text("battle.victory") == "Victory"
    
    i18n.set_language("zh-CN")
    assert i18n.get_text("menu.settings") == "设置"
    assert i18n.get_text("battle.victory") == "胜利"

def test_fallback_behavior(i18n):
    """Test fallback to default language when translation is missing."""
    i18n.set_language("zh-CN")
    # Use a key that doesn't exist in Chinese but exists in English
    assert i18n.get_text("nonexistent.key", default="Fallback") == "Fallback"
    
    # Test fallback to key itself when no default provided
    assert i18n.get_text("nonexistent.key") == "nonexistent.key"

def test_supported_languages(i18n):
    """Test supported languages list."""
    languages = i18n.get_supported_languages()
    assert isinstance(languages, list)
    assert len(languages) == 3
    assert all(lang in ["en-US", "zh-CN", "ja-JP"] for lang in languages)

def test_current_language(i18n):
    """Test current language getter."""
    assert i18n.get_current_language() == "en-US"
    i18n.set_language("zh-CN")
    assert i18n.get_current_language() == "zh-CN"

def test_reload_translations(i18n):
    """Test translation reloading."""
    original_translations = i18n.translations.copy()
    i18n.reload_translations()
    assert i18n.translations == original_translations

@pytest.mark.parametrize("language,key,expected", [
    ("en-US", "common.yes", "Yes"),
    ("zh-CN", "common.yes", "是"),
    ("ja-JP", "common.yes", "はい"),
    ("en-US", "common.no", "No"),
    ("zh-CN", "common.no", "否"),
    ("ja-JP", "common.no", "いいえ"),
])
def test_specific_translations(i18n, language, key, expected):
    """Test specific translations across different languages."""
    i18n.set_language(language)
    assert i18n.get_text(key) == expected
