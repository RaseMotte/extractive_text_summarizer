import time

def timerfunc(func, logger):
    def function_timer(*args, **kwargs):
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        msg = "Runtime for {func} : {time} seconds"
        logger.info(msg.format(func=func.__name__, time=runtime))
        return value
    return function_timer

class log_time(object):
    def __init__(self, logger):
        self.decorator = timerfunc
        self.logger = logger

    def __call__(self, func):
        return self.decorator(func, self.logger)