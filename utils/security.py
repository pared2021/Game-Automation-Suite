import re
from cryptography.fernet import Fernet
from utils.error_handler import log_exception

class SecurityManager:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)

    @log_exception
    def validate_input(self, input_string, pattern):
        # 验证输入字符串是否匹配指定的模式
        return bool(re.match(pattern, input_string))

    @log_exception
    def encrypt_data(self, data):
        # 加密数据
        return self.cipher_suite.encrypt(data.encode()).decode()

    @log_exception
    def decrypt_data(self, encrypted_data):
        # 解密数据
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()

security_manager = SecurityManager()