import logging
import os

LOG_DIR = '../../model'

class ModelLogger:
    def __init__(self, model_name):

        self.logger = logging.getLogger("ModelLogger")
        self.logger.setLevel(logging.INFO)

        model_data_path = os.path.join(LOG_DIR, model_name)
        if not os.path.exists(model_data_path):
            os.makedirs(model_data_path)

        info_log_path = os.path.join(model_data_path, '%s.log' % model_name)
        rouge_log_path = os.path.join(model_data_path, '%s_rouge.log' % model_name)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(info_log_path)
        fhd = logging.FileHandler(rouge_log_path)
        fh.setLevel(logging.INFO)
        fhd.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        #ch = logging.StreamHandler()
        #ch.setLevel(logging.ERROR)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        #ch.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(fhd)
        #self.logger.addHandler(ch)

    def info(self, msg):
        self.logger.info(msg)

    def rouge_debug(self, bin_path, record_i, msg):
        json_msg = '{"data_path" : %s, "record_id" : %d, "metrics": %s}' % (bin_path, record_i, msg)
        self.logger.debug(json_msg)

    def error(self, msg):
        self.logger.info(msg)
