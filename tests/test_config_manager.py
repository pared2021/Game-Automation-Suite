import os
import pytest
from game_automation.config.config_manager import ConfigManager

@pytest.fixture
def config_manager():
    """Fixture to provide a clean ConfigManager instance"""
    return ConfigManager()

def test_config_loading(config_manager):
    """Test that configurations are loaded correctly"""
    # Verify that required configs are loaded
    assert 'game_settings' in config_manager.config
    assert 'resource_paths' in config_manager.config
    
    # Verify specific values
    assert isinstance(config_manager.config['game_settings'], dict)
    assert isinstance(config_manager.config['resource_paths'], dict)

def test_get_method(config_manager):
    """Test the get() method"""
    # Test existing key
    assert isinstance(config_manager.get('game_settings'), dict)
    
    # Test non-existent key
    with pytest.raises(KeyError):
        config_manager.get('non_existent_key')

def test_get_with_default(config_manager):
    """Test the get_with_default() method"""
    # Test existing key
    assert isinstance(config_manager.get_with_default('game_settings', {}), dict)
    
    # Test non-existent key with default
    assert config_manager.get_with_default('non_existent_key', 'default') == 'default'

def test_set_method(config_manager, tmpdir):
    """Test the set() method"""
    # Test setting a new value
    config_manager.set('test_key', 'test_value')
    assert config_manager.get('test_key') == 'test_value'
    
    # Test saving to file
    test_file = tmpdir.join('test_config.yaml')
    config_manager.set('test_key', 'test_value', save_to_file=True)
    assert os.path.exists(str(test_file))

def test_validate_configs(config_manager):
    """Test the validate_configs() method"""
    # Test valid configs
    is_valid, errors = config_manager.validate_configs()
    assert is_valid
    assert len(errors) == 0
    
    # Test invalid configs
    config_manager.config['game_settings']['battle']['auto_battle']['strategy'] = 'invalid'
    is_valid, errors = config_manager.validate_configs()
    assert not is_valid
    assert len(errors) > 0

def test_save_config(config_manager, tmpdir):
    """Test the save_config() method"""
    # Create a test config file
    test_file = tmpdir.join('test_config.yaml')
    with open(test_file, 'w') as f:
        f.write('test_key: test_value\n')
    
    # Modify and save config
    config_manager.set('test_key', 'new_value')
    config_manager.save_config('test_key')
    
    # Verify the change was saved
    with open(test_file, 'r') as f:
        content = f.read()
        assert 'new_value' in content
