import os
import yaml
from typing import Dict, Optional, Any
from pathlib import Path

class I18nManager:
    """Internationalization manager for handling multiple languages."""
    
    def __init__(self, default_language: str = "en-US"):
        """Initialize the I18n manager.
        
        Args:
            default_language (str): Default language code (e.g., 'en-US')
        """
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, Any]] = {}
        self.supported_languages = ["en-US", "zh-CN", "ja-JP"]
        
        # Load all translations
        self._load_translations()
    
    def _load_translations(self) -> None:
        """Load all translation files from the locales directory."""
        locale_dir = Path(__file__).parent / "locales"
        
        for lang in self.supported_languages:
            file_path = locale_dir / f"{lang}.yaml"
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.translations[lang] = yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading translation file {lang}: {e}")
                # Initialize empty translation if file loading fails
                self.translations[lang] = {}
    
    def set_language(self, language: str) -> bool:
        """Set the current language.
        
        Args:
            language (str): Language code to set
            
        Returns:
            bool: True if language was set successfully, False otherwise
        """
        if language in self.supported_languages:
            self.current_language = language
            return True
        return False
    
    def get_text(self, key: str, default: Optional[str] = None, **kwargs) -> str:
        """Get translated text for a given key.
        
        Args:
            key (str): Translation key (e.g., 'common.ok')
            default (Optional[str]): Default text if key not found
            **kwargs: Format arguments for the translated string
            
        Returns:
            str: Translated text
        """
        # Split the key into parts (e.g., 'common.ok' -> ['common', 'ok'])
        parts = key.split('.')
        
        # Try to get translation from current language
        value = self._get_nested_value(self.translations.get(self.current_language, {}), parts)
        
        # Fallback to default language if translation not found
        if value is None and self.current_language != self.default_language:
            value = self._get_nested_value(self.translations.get(self.default_language, {}), parts)
        
        # Use default value or key itself if translation not found
        if value is None:
            value = default if default is not None else key
        
        # Format string with provided arguments
        try:
            return value.format(**kwargs) if kwargs else value
        except (KeyError, ValueError):
            return value
    
    def _get_nested_value(self, data: Dict[str, Any], keys: list) -> Optional[str]:
        """Get a nested value from a dictionary using a list of keys.
        
        Args:
            data (Dict[str, Any]): Dictionary to search in
            keys (list): List of keys to traverse
            
        Returns:
            Optional[str]: Found value or None if not found
        """
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key)
            else:
                return None
        return data if isinstance(data, str) else None
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages.
        
        Returns:
            list: List of supported language codes
        """
        return self.supported_languages
    
    def get_current_language(self) -> str:
        """Get current language code.
        
        Returns:
            str: Current language code
        """
        return self.current_language
    
    def reload_translations(self) -> None:
        """Reload all translation files."""
        self._load_translations()

# Create a global instance
i18n = I18nManager()

# Example usage:
if __name__ == "__main__":
    # Get translation
    print(i18n.get_text("common.ok"))  # Output: OK
    
    # Switch language
    i18n.set_language("zh-CN")
    print(i18n.get_text("common.ok"))  # Output: 确定
    
    # Use format arguments
    print(i18n.get_text("character.level"))  # Output: 等级
    
    # Get supported languages
    print(i18n.get_supported_languages())  # Output: ['en-US', 'zh-CN', 'ja-JP']
