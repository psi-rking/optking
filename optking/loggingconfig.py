import os
import sys

logging_configuration = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "time_severity_message": {
            "format": "[{asctime}] - [{levelname}]: {message}",
            "style": "{"
            },
        "severity_name_message": {
            "format": "[{levelname}] [{name}]: {message}",
            "style": "{"
            },
        "severity_message":{
            "format": "[{levelname}]: {message}",
            "style": "{"
        },
        "message_format": {
            "format": "{message}",
            "style": "{"
                }
    },
    "handlers": {
        "terminal_debug": {
            "class": "logging.StreamHandler",
            "formatter": "severity_message",
            "level": "DEBUG"
        },
        "terminal_info": {
            "class": "logging.StreamHandler",
            "formatter": "severity_message",
            "level": "INFO"
        },
        "file_log_debug": {
            "class": "logging.FileHandler",
            "formatter": "severity_message",
            "level": "DEBUG",
            "filename": os.path.join(os.getcwd(), 'opt_log.out')
        },
        "file_log_info": {
            "class": "logging.FileHandler",
            "mode": "a",
            "formatter": "severity_message",
            "level": "INFO",
            "filename": os.path.join(os.getcwd(), 'opt_log.out')
        },
        "file_king_info": {
            "class": "logging.FileHandler",
            "formatter": "message_format",
            "level": "INFO",
            "filename": os.path.join(os.getcwd(), 'opt_log.out')
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["file_log_debug"]
        
    }
}
