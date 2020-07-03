import logging
import time


def config_log():
    c_time = str(time.strftime('%Y%m%d %H.%M', time.localtime(time.time())))
    log_file = 'log/log_' + c_time + ".log"
    formatter1 = logging.Formatter('%(asctime)s:  %(message)s')
    formatter2 = logging.Formatter('%(levelname)s: %(message)s')

    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel('DEBUG')
    f_handler.setFormatter(formatter1)

    c_handler = logging.StreamHandler()
    c_handler.setLevel('DEBUG')
    c_handler.setFormatter(formatter2)

    my_logger = logging.getLogger('mylogger')
    my_logger.setLevel(logging.DEBUG)
    my_logger.addHandler(f_handler)
    my_logger.addHandler(c_handler)
    return my_logger


logger = config_log()
