import os
import sys

logging_configuration = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "time_severity_message": {
            "format": "%(asctime)s - %(levelname) - %(message)s"
            },
        "severity_message": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
            },
        "message_format": {
            "format": "%(message)"
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
            "filename": os.path.join(os.getcwd(), sys.argv[0][:-3]) + ".out"
        },
        "file_log_info": {
            "class": "logging.FileHandler",
            "mode": "a",
            "formatter": "severity_message",
            "level": "INFO",
            "filename": os.path.join(os.getcwd(), sys.argv[0][:-3]) + ".out"
        },
        "file_king_info": {
            "class": "logging.FileHandler",
            "formatter": "message_format",
            "level": "INFO",
            "filename": os.path.join(os.getcwd(), sys.argv[0][:-3]) + ".out"
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["file_log_debug"]
    }
}
