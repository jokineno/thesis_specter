import logging

def setup_logging(log_output=None):
    if log_output is None:
        handlers = [
            logging.StreamHandler()
        ]
    else:
        handlers = [
            logging.FileHandler(log_output),
            logging.StreamHandler()
        ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )
    return logging.getLogger()


logger = setup_logging()
