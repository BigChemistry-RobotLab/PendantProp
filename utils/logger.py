import logging
import os

class Logger:
    def __init__(self, name: str, file_path: str):
        os.environ["NUMEXPR_MAX_THREADS"] = (
            "8"  # set the number of threads for numexpr (avoid logger warning)
        )
        self.name = name
        self.file_path = file_path
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Check if the logger already has handlers to avoid duplicate logs
        if not self.logger.handlers:
            # Create file handler
            file_handler = logging.FileHandler(f"{self.file_path}/{self.name}.log")
            file_handler.setLevel(logging.INFO)

            # Create stream handler
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)

            # Create formatter and add it to the handlers
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)

            # Add handlers to the logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def __str__(self):
        return f"""
        Logger object
        Name: {self.name}
        Log file path: {self.file_path}
        """
