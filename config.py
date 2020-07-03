
def get_run_config(data_name):
    n_epoch = 30
    lr = 0.1

    if data_name in ["annthyroid"]:
        n_epoch = 80

    if data_name in ["ad"]:
        lr = 0.001

    return n_epoch, lr
