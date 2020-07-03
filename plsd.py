"""
@author: Hongzuo XU

"""

import numpy as np
import utils
import torch
import classification
from numpy.random import RandomState
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from Net import MLPDrop, MLPDrop2
from log import logger


class PLSD:
    def __init__(self, device, name, use_nor, epoch_per_step, n_hidden,
                 drop_prob, batch_size="auto", n_epochs=30, lr=0.1, seed=-1, verbose=True):
        # data
        self.dataset_name = name
        self.x_train = None
        self.x_test = None
        self.dimension = 0
        self.n_x = 0

        # config
        self.device = device
        if seed == -1:
            self.seed = None
        else:
            self.seed = seed
        self.verbose = verbose

        # Parameters of PLSD
        self.use_nor = use_nor
        self.epoch_per_step = epoch_per_step

        self.n_hidden = n_hidden
        self.drop_prob = drop_prob
        if batch_size == "auto":
            self.batch_size = -1
        else:
            self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr

        # middle variables
        self.init_ano_ids = None
        self.init_nor_ids = None
        self.n_init_ano = 0
        self.n_init_nor = 0

        self.ano_ids = None
        self.nor_ids = None
        self.init_score = None

        self.surrogate_x = None
        self.surrogate_y = None

        self.net = None
        self.optimizer1 = None
        self.loss_func1 = None

        self.inlier_efficacy = None

    def fit(self, x_train, semi_y_train):
        self.x_train = x_train
        self.dimension = x_train.shape[1]
        self.n_x = x_train.shape[0]

        # # calculate initial anomaly scores
        self.init_score = self.calc_init_score()

        # # # identify reliable and diversified inliers & known anomalies
        self.init_ano_ids = np.where(semi_y_train == 1)[0]
        self.init_nor_ids = self.explore_inliers()

        self.nor_ids = self.init_nor_ids.copy()
        self.ano_ids = self.init_ano_ids.copy()

        self.n_init_ano = len(self.ano_ids)
        self.n_init_nor = len(self.nor_ids)

        # # $$$ ablation study to test the effective of heuristic inlier sampling
        # # $$$ replace heuristic sampling with random sampling
        # size = len(self.init_nor_ids)
        # self.init_nor_ids = RandomState(seed=None).choice(np.where(semi_y_train == 0)[0], size, replace=False)
        # self.nor_ids = self.init_nor_ids.copy()
        # logger.info("ABLATION STUDY: replace heuristic sampling with random sampling using same size")

        labeled_size = len(self.nor_ids) + len(self.ano_ids)
        logger.info("init labeled size: %d (%d anomaly + %d normal)" %
                    (labeled_size, len(self.ano_ids), len(self.nor_ids)))

        self.iter_deviation_learning(epoch_per_step=self.epoch_per_step)
        return

    def predict(self, x_test):
        n_test = len(x_test)
        n_selected_inl = self.use_nor
        # select high-efficacy inliers
        chosen = utils.get_sorted_index(self.inlier_efficacy)[:n_selected_inl]
        use_inlier_indices = self.nor_ids[chosen]
        use_inlier = self.x_train[use_inlier_indices]
        n_use_inlier = len(use_inlier_indices)

        # combine with selected high-efficacy inliers and feed into network
        combined = np.zeros([n_test * n_use_inlier, self.dimension * 2])
        for ii, x in enumerate(x_test):
            for jj, normal in enumerate(use_inlier):
                combined[ii * n_use_inlier + jj] = np.concatenate((x, normal), axis=0)

        # _, y_prob = classification.pure_predict(combined, self.net, self.device)
        _, y_prob, out_dic = classification.pure_predict_full_out(combined, self.net, self.device)

        weight = self.inlier_efficacy[chosen]
        weight = utils.sum_norm(weight)

        anomaly_score = np.zeros(n_test)
        for i in range(n_test):
            this_prob = y_prob[i * n_use_inlier: (i + 1) * n_use_inlier]
            score = (this_prob[:, 1]) * weight
            anomaly_score[i] = np.sum(score)
        return anomaly_score, out_dic

    def iter_deviation_learning(self, epoch_per_step):
        n_input1 = self.dimension * 2
        # self.net = MLPDrop(n_input=n_input1, n_hidden=self.n_hidden, n_output=2, drop_p=self.drop_prob).to(self.device)
        self.net = MLPDrop2(n_input=n_input1, n_hidden=self.n_hidden, n_output=2, drop_p=self.drop_prob).to(self.device)
        self.optimizer1 = torch.optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        self.loss_func1 = torch.nn.CrossEntropyLoss()

        iter_count = 1
        max_iter = int(self.n_epochs / epoch_per_step)
        while True:
            surrogate_x, surrogate_y = self.generate_surrogate_supervision()
            surrogate_x_ba, surrogate_y_ba, flag = self.get_balanced_data(surrogate_x, surrogate_y)
            logger.info("#%d | iter train data size: %d; by %s" % (iter_count, len(surrogate_y_ba), flag))

            self.nn_classifier_train(surrogate_x_ba, surrogate_y_ba, n_epoch=epoch_per_step)
            temp_score = self.nn_classifier_scoring(surrogate_x, surrogate_y, self.ano_ids, self.nor_ids)

            new_ano_ids, new_nor_ids = self.extend_labeled_data(temp_score)
            self.ano_ids = np.append(self.ano_ids, new_ano_ids)
            self.nor_ids = np.append(self.nor_ids, new_nor_ids)

            if iter_count == max_iter or len(new_ano_ids) + len(new_nor_ids) == 0:
                break
            iter_count += 1

        remain_epoch = max(self.n_epochs - iter_count * epoch_per_step, 0)
        if remain_epoch > 0:
            surrogate_x, surrogate_y = self.generate_surrogate_supervision()
            surrogate_x_ba, surrogate_y_ba, flag = self.get_balanced_data(surrogate_x, surrogate_y)
            logger.info("## | iter train data size: %d;  by %s" % (len(surrogate_y_ba), flag))
            self.nn_classifier_train(surrogate_x_ba, surrogate_y_ba, n_epoch=remain_epoch)
        return

    def nn_classifier_train(self, x, y, n_epoch):
        # n_step per epoch is controlled to be smaller than 1000
        if self.batch_size == -1:
            if len(y) > 256000:
                each_batch_size = 512
            elif len(y) > 128000:
                each_batch_size = 256
            elif len(y) > 64000:
                each_batch_size = 128
            else:
                each_batch_size = 64
        else:
            each_batch_size = self.batch_size
        self.net = classification.train(
            x=x, y=y, net=self.net, optimizer=self.optimizer1, loss_func=self.loss_func1,
            device=self.device, batch_size=each_batch_size, n_epochs=n_epoch, verbose=self.verbose
        )

    def nn_classifier_scoring(self, x, y, ano_ids, nor_ids):
        self.inlier_efficacy = classification.calc_efficacy(x=x, y=y, net=self.net, device=self.device,
                                                            ano_size=len(ano_ids), nor_size=len(nor_ids))

        n_use_inlier = self.use_nor
        chosen = utils.get_sorted_index(self.inlier_efficacy)[:n_use_inlier]
        use_inlier_indices = nor_ids[chosen]
        use_inlier = self.x_train[use_inlier_indices]

        n_x = self.n_x
        combined_x = np.zeros([n_x * n_use_inlier, self.dimension * 2])
        for ii, x in enumerate(self.x_train):
            for jj, normal in enumerate(use_inlier):
                combined_x[ii * n_use_inlier + jj] = np.concatenate((x, normal), axis=0)
        y_pred, y_prob = classification.pure_predict(combined_x, self.net, self.device)

        weight = self.inlier_efficacy[chosen]
        weight = utils.sum_norm(weight)
        anomaly_score = np.zeros(n_x)
        for i in range(n_x):
            this_prob = y_prob[i * n_use_inlier: (i + 1) * n_use_inlier]
            score = (this_prob[:, 1]) * weight
            anomaly_score[i] = np.sum(score)
        return anomaly_score

    def get_balanced_data(self, surrogate_x, surrogate_y):
        if len(surrogate_y) < len(self.x_train) or len(np.where(surrogate_y == 1)[0]) < 500:
            model = RandomOverSampler(random_state=self.seed)
            surrogate_x_ba, surrogate_y_ba = model.fit_resample(surrogate_x, surrogate_y)
            flag = "over sampling"
        else:
            model = RandomUnderSampler(random_state=self.seed)
            surrogate_x_ba, surrogate_y_ba = model.fit_resample(surrogate_x, surrogate_y)
            flag = "under sampling"
        return surrogate_x_ba, surrogate_y_ba, flag

    def extend_labeled_data(self, anomaly_score):
        ano_ids = self.ano_ids
        nor_ids = self.nor_ids
        known_ano_score = anomaly_score[ano_ids]
        known_nor_score = anomaly_score[nor_ids]

        ano_range = [min(known_ano_score), max(known_ano_score)]
        new_ano_ids = np.where(anomaly_score >= ano_range[1])[0]
        nor_range = [min(known_nor_score), max(known_nor_score)]
        new_nor_ids = np.where(anomaly_score <= nor_range[0])[0]

        new_ano_ids = np.setdiff1d(new_ano_ids, np.intersect1d(ano_ids, new_ano_ids))
        new_nor_ids = np.setdiff1d(new_nor_ids, np.intersect1d(nor_ids, new_nor_ids))

        if len(new_nor_ids) > 5000:
            # anomaly_score = self.init_score[new_nor_ids]
            sampling_prob = (1 - anomaly_score) / np.sum(1 - anomaly_score)
            n = 5000
            new_nor_ids = RandomState(seed=self.seed).choice(new_nor_ids, n, p=sampling_prob, replace=False)
            logger.debug("restrict the number of explored normal data to 5000")

        return new_ano_ids, new_nor_ids

    def calc_init_score(self):
        mean = np.average(self.x_train, axis=0)
        std = np.std(self.x_train, axis=0)
        zero_ids = np.where(std == 0)[0]
        if len(zero_ids) > 0:
            logger.warning("! x_train with feature(s) having zero std.")
            std[zero_ids] = 1

        init_score = np.array([np.average(((x - mean) / std) ** 2) for x in self.x_train])
        # init_score = np.array([np.average((x - mean) / std) for x in self.x_train])
        init_score = utils.min_max_norm(init_score)
        return init_score

    def explore_inliers(self):
        N = self.n_x
        init_score = self.init_score

        n_known_anomaly = len(self.init_ano_ids)
        size = n_known_anomaly * 10
        n_sampling = max(min(200, size), n_known_anomaly)
        # n_sampling = max(min(200, 10*n_known_anomaly), n_known_anomaly)
        candidate_indices = utils.get_sorted_index(init_score)[-int(0.5 * N):]
        candidate_indices = np.delete(candidate_indices, self.init_ano_ids)
        anomaly_score = init_score[candidate_indices]
        sampling_prob = (1 - anomaly_score) / np.sum(1 - anomaly_score)
        if n_sampling > len(candidate_indices):
            n_sampling = len(candidate_indices)

        inlier_indices = RandomState(seed=self.seed).choice(candidate_indices, n_sampling,
                                                            p=sampling_prob, replace=False)
        return inlier_indices

    def generate_surrogate_supervision(self):
        x = self.x_train
        dim = self.dimension
        ano_indices = self.ano_ids
        nor_indices = self.nor_ids

        labeled_x = np.concatenate((x[ano_indices], x[nor_indices]), axis=0)
        labeled_y = np.append(np.ones(len(ano_indices), dtype=int), np.zeros(len(nor_indices), dtype=int))

        ano_size, nor_size = len(ano_indices), len(nor_indices)
        labeled_size = ano_size + nor_size

        ss_size = ano_size * nor_size + int((nor_size * (nor_size - 1)) * 0.5)
        surrogate_x = np.zeros([ss_size, dim * 2])
        surrogate_y = np.zeros([ss_size], dtype=int)

        count = 0
        for i in range(labeled_size):
            for j in range(i + 1, labeled_size):
                if labeled_y[i] + labeled_y[j] == 2:
                    continue
                surrogate_x[count] = np.append(labeled_x[i], labeled_x[j])
                surrogate_y[count] = labeled_y[i] + labeled_y[j]
                count += 1

        return surrogate_x, surrogate_y
