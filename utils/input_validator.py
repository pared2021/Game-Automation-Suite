import re
from utils.error_handler import InputError

class InputValidator:
    @staticmethod
    def validate_item_name(item_name):
        if not re.match(r'^[a-zA-Z0-9_]+$', item_name):
            raise InputError(f"Invalid item name: {item_name}")
        return item_name

    @staticmethod
    def validate_task_id(task_id):
        try:
            task_id = int(task_id)
            if task_id < 0:
                raise ValueError
            return task_id
        except ValueError:
            raise InputError(f"Invalid task ID: {task_id}")

    @staticmethod
    def validate_resource_name(resource_name):
        if not re.match(r'^[a-zA-Z0-9_]+$', resource_name):
            raise InputError(f"Invalid resource name: {resource_name}")
        return resource_name

    @staticmethod
    def sanitize_input(input_string):
        # Remove any potentially dangerous characters
        return re.sub(r'[^\w\s-]', '', input_string).strip()

input_validator = InputValidator()