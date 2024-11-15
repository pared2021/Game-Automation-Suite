import os
import pytest
import shutil
import tempfile
from pathlib import Path

@pytest.fixture(scope="session")
def test_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def base_dir():
    """Return the project base directory."""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def config_dir(base_dir):
    """Return the configuration directory."""
    return base_dir / "config"

@pytest.fixture(scope="session")
def resource_dir(base_dir):
    """Return the resources directory."""
    return base_dir / "resources"

@pytest.fixture(scope="session")
def i18n_dir(base_dir):
    """Return the i18n directory."""
    return base_dir / "game_automation" / "i18n"

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, test_dir):
    """Set up test environment variables."""
    monkeypatch.setenv("GAME_AUTOMATION_ENV", "test")
    monkeypatch.setenv("GAME_AUTOMATION_CONFIG_DIR", str(Path(test_dir) / "config"))
    monkeypatch.setenv("GAME_AUTOMATION_RESOURCE_DIR", str(Path(test_dir) / "resources"))
    
    # Create necessary test directories
    os.makedirs(os.path.join(test_dir, "config"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "resources"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "logs"), exist_ok=True)

@pytest.fixture
def sample_config():
    """Return a sample configuration for testing."""
    return {
        "game_settings": {
            "difficulty": "normal",
            "max_players": 4
        },
        "resource_paths": {
            "images": "./resources/images",
            "sounds": "./resources/sounds"
        },
        "i18n": {
            "default_language": "en-US",
            "supported_languages": ["en-US", "zh-CN", "ja-JP"]
        }
    }

@pytest.fixture
def sample_game_settings():
    """Return sample game settings for testing."""
    return {
        "battle": {
            "auto_battle": {
                "enabled": True,
                "strategy": "balanced"
            }
        },
        "character": {
            "auto_level_up": True,
            "auto_equip": True
        }
    }

@pytest.fixture
def mock_file_structure(test_dir):
    """Create a mock file structure for testing."""
    # Create directories
    dirs = [
        "config",
        "resources/images",
        "resources/sounds",
        "logs",
        "game_automation/i18n/locales"
    ]
    for d in dirs:
        os.makedirs(os.path.join(test_dir, d), exist_ok=True)
    
    # Create sample files
    files = {
        "config/config.yaml": "# Test config",
        "config/game_settings.yaml": "# Test game settings",
        "logs/test.log": "# Test log"
    }
    for path, content in files.items():
        with open(os.path.join(test_dir, path), 'w', encoding='utf-8') as f:
            f.write(content)
    
    return test_dir

@pytest.fixture
def cleanup_files():
    """Cleanup any test files after tests."""
    yield
    patterns = ["*.log", "*.tmp", "*.test"]
    test_dir = os.path.dirname(__file__)
    for pattern in patterns:
        for file in Path(test_dir).glob(pattern):
            try:
                file.unlink()
            except OSError:
                pass

@pytest.fixture
def mock_i18n_files(test_dir):
    """Create mock i18n files for testing."""
    i18n_dir = os.path.join(test_dir, "game_automation/i18n/locales")
    os.makedirs(i18n_dir, exist_ok=True)
    
    # Create sample translation files
    translations = {
        "en-US.yaml": """
common:
  yes: "Yes"
  no: "No"
  ok: "OK"
""",
        "zh-CN.yaml": """
common:
  yes: "是"
  no: "否"
  ok: "确定"
""",
        "ja-JP.yaml": """
common:
  yes: "はい"
  no: "いいえ"
  ok: "OK"
"""
    }
    
    for filename, content in translations.items():
        with open(os.path.join(i18n_dir, filename), 'w', encoding='utf-8') as f:
            f.write(content)
    
    return i18n_dir
