import logging
from functools import wraps
import traceback
import sentry_sdk
from utils.config_manager import config_manager

sentry_sdk.init(
    dsn=config_manager.get('error_reporting.sentry_dsn'),
    traces_sample_rate=1.0
)

class GameAutomationError(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(GameAutomationError):
    """Exception raised for errors in the input."""
    pass

class DeviceError(GameAutomationError):
    """Exception raised for errors related to the device."""
    pass

class OCRError(GameAutomationError):
    """Exception raised for errors in OCR processing."""
    pass

class NetworkError(GameAutomationError):
    """Exception raised for network-related errors."""
    pass

def log_exception(logger):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except GameAutomationError as e:
                logger.error(f"{type(e).__name__} in {func.__name__}: {str(e)}")
                sentry_sdk.capture_exception(e)
                raise
            except Exception as e:
                logger.critical(f"Unexpected {type(e).__name__} in {func.__name__}: {str(e)}")
                logger.critical(traceback.format_exc())
                sentry_sdk.capture_exception(e)
                raise GameAutomationError(f"An unexpected error occurred: {str(e)}")
        return wrapper
    return decorator

def handle_errors(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except InputError as e:
            logging.warning(f"Input error: {str(e)}")
            # Handle input errors (e.g., request user to correct input)
        except DeviceError as e:
            logging.error(f"Device error: {str(e)}")
            # Handle device errors (e.g., attempt to reconnect or restart device)
        except OCRError as e:
            logging.error(f"OCR error: {str(e)}")
            # Handle OCR errors (e.g., retry OCR or use alternative method)
        except NetworkError as e:
            logging.error(f"Network error: {str(e)}")
            # Handle network errors (e.g., retry connection or switch to offline mode)
        except GameAutomationError as e:
            logging.error(f"Game automation error: {str(e)}")
            # Handle general game automation errors
        except Exception as e:
            logging.critical(f"Unexpected error: {str(e)}")
            logging.critical(traceback.format_exc())
            sentry_sdk.capture_exception(e)
            # Handle unexpected errors (e.g., graceful shutdown or restart)
    return wrapper

def report_error(error, context=None):
    sentry_sdk.capture_exception(error)
    if context:
        sentry_sdk.set_context("error_context", context)
    logging.error(f"Reported error to Sentry: {str(error)}")

# Example usage:
# @handle_errors
# async def some_function():
#     # Function implementation
#     pass