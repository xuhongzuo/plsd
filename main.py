"""
@author: Hongzuo XU
This algorithm is implemented via Python 3.7 and torch 1.2.
This is the source code of PLSD, a weakly-supervised anomaly detection method with partially labeled anomalies.
"""

import numpy as np
import pandas as pd
import torch
import time
import os

import utils
import config
from plsd import PLSD
from log import logger


torch.set_num_threads(1)

remark = "@PLSD"
doc = open('zout.txt', 'a')
print(remark, file=doc)
doc.close()


def main(file_path, use_nor, epoch_per_step, batch_size="auto", n_epoch=30, lr=0.1, seed=-1,
         test_percentage=0.4, ratio=0.1, n_run=10):

    data_name = file_path.split("/")[-1].split(".")[0]

    df = pd.read_csv(file_path)
    x = df.values[:, :-1]
    labels = np.array(df.values[:, -1], dtype=int)
    dim = x.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if dim > 500 else "cpu"
    logger.debug("Device, %s\n" % device)

    n_hidden = int(2.5 * dim)
    drop_prob = 0.2

    logger.info("-------- DATASET: %s, Known_ratio: %.2f, Runs: %d --------" % (data_name, ratio, n_run))

    r_auc_roc, r_auc_pr, r_time = np.zeros(n_run), np.zeros(n_run), np.zeros(n_run)
    for i in range(n_run):
        x_train, y_train, x_test, y_test = utils.split_train_test(x, labels, test_size=test_percentage, seed=seed)
        semi_y_train = utils.semi_setting(x_train, y_train, ratio, seed)

        n_anomalies = len(np.where(y_train == 1)[0])
        n_k_anomalies = len(np.where(semi_y_train == 1)[0])
        n_test_anomalies = len(np.where(y_test == 1)[0])
        if i == 0:
            logger.info("@@@ TrainData Shape: %d (%d) * %d " % (x_train.shape[0], n_anomalies, x_train.shape[1]))
            logger.info("@@@ TestData Shape: %d (%d) * %d " % (x_test.shape[0], n_test_anomalies, x_test.shape[1]))

            logger.info("@@@ Train Known/True Anomalies: %d/%d" % (n_k_anomalies, n_anomalies))
            logger.info("@@@ Net Structure: input %d, hidden1 %d, out %d, drop prob %.1f" %
                        (2*dim, n_hidden, 2, drop_prob))

        s_time = time.time()
        plsd = PLSD(device=device, name=data_name, use_nor=use_nor, epoch_per_step=epoch_per_step,
                    n_hidden=n_hidden, drop_prob=0.2, batch_size=batch_size, n_epochs=n_epoch, lr=lr,
                    seed=seed, verbose=False)
        plsd.fit(x_train, semi_y_train)
        y_score, out_dic = plsd.predict(x_test)

        r_time[i] = time.time()-s_time
        au_roc, au_pr = utils.get_performance(y_score, y_test)
        r_auc_roc[i] = round(au_roc, 3) * 100
        r_auc_pr[i] = round(au_pr, 3) * 100
        logger.info("-----%s, Round %d: AUC-ROC: %.1f, AUC-PR: %.1f, %.4fs\n" %
                    (data_name, i+1, r_auc_roc[i], r_auc_pr[i], r_time[i]))

    txt = data_name + ", AUC-ROC, %.1f, %.1f , AUC-PR, %.1f, %.1f, %.1fs, %druns, %.2f, PLSD" % \
          (np.average(r_auc_roc), np.std(r_auc_roc), np.average(r_auc_pr), np.std(r_auc_pr),
           np.average(r_time), runs, ratio)
    logger.info(txt)
    doc = open('zout.txt', 'a')
    print(txt, file=doc)
    doc.close()
    return


if __name__ == '__main__':
    input_root_list = ["data/aYahoo.csv"]
    # input_root_list = ["data/synthetic/"]
    # input_root_list = ["data/scal_test/"]

    runs = 10
    seed = -1
    default_test_p = 0.4
    default_r = 0.1
    r_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    use_n = 10
    epoch_per_s = 10
    bs = "auto"

    for input_root in input_root_list:
        if os.path.isdir(input_root):
            for file_name in sorted(os.listdir(input_root)):
                if file_name.endswith(".csv"):
                    input_path = str(os.path.join(input_root, file_name))
                    name = input_path.split("/")[-1].split('.')[0]
                    epoch, lr = config.get_run_config(name)
                    main(input_path, use_nor=use_n, epoch_per_step=epoch_per_s, batch_size=bs, n_epoch=epoch, lr=lr,
                         seed=seed, test_percentage=default_test_p, ratio=default_r, n_run=runs)

                    # for r in r_list:
                    #     main(input_path, use_nor=use_n, epoch_per_step=epoch_per_s,
                    #          batch_size=bs, n_epoch=epoch, lr=lr,
                    #          seed=rand_seed, test_percentage=default_test_p, ratio=r, n_run=runs)

        else:
            input_path = input_root
            name = input_path.split("/")[-1].split(".")[0]

            epoch, lr = config.get_run_config(name)
            main(input_path, use_nor=use_n, epoch_per_step=epoch_per_s, batch_size=bs, n_epoch=epoch, lr=lr,
                 seed=seed, test_percentage=default_test_p, ratio=default_r, n_run=runs)
