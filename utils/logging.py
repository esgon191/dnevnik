import logging

def logger_factory(name : str, file : str, level=logging.DEBUG, mode='a'):
    logger = logging.getLogger(name=name)

    handler = logging.FileHandler(file, mode=mode)
    handler.setLevel(level)

    data_generator_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(data_generator_formatter)

    logger.addHandler(handler)

    return logger