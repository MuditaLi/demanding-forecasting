import logging


def format_message(message_body: str, message_parameter: str = None, class_name: str = None):
    message = message_body
    if message_parameter is not None:
        message = ' '.join([(message + ' ').ljust(50, '.'), message_parameter])

    if class_name is not None:
        message = ' - '.join([class_name.upper(), message])

    return message


class Logger:
    # create logger
    logger = logging.getLogger(__name__)

    if not logger.handlers:
        logger.propagate = False  # Prevent from generating one output per module
        logger.setLevel(logging.DEBUG)

        # create_file_handlers
        file_handler = logging.FileHandler(filename='logs.log', mode='a')  # Append to existing logs
        file_handler.setLevel(logging.DEBUG)

        file_handler_run = logging.FileHandler(filename='logs_run.log', mode='w')  # Create new run-specific logs
        file_handler_run.setLevel(logging.DEBUG)

        # create console handler and set level to debug & add console_handler to logger
        console_handler = logging.StreamHandler()  # (sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # add formatter to console_handler
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        file_handler_run.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.addHandler(file_handler_run)

    @staticmethod
    def debug(message_body: str, message_parameter: str=None, class_name: str=None):
        Logger.logger.debug(format_message(message_body, message_parameter, class_name))

    @staticmethod
    def warning(message_body: str, message_parameter: str = None, class_name: str = None):
        Logger.logger.warning(format_message(message_body, message_parameter, class_name))

    @staticmethod
    def info(message_body: str, message_parameter: str = None, class_name: str = None):
        Logger.logger.info(format_message(message_body, message_parameter, class_name))

    @staticmethod
    def test_messages():
        # 'application' code
        Logger.logger.debug('debug message')
        Logger.logger.info('info message')
        Logger.logger.warning('warn message')
        Logger.logger.error('error message')
        Logger.logger.critical('critical message')
