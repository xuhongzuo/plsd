import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from log import logger


def split_train_test(x, y, test_size, seed):
    idx_norm = y == 0
    idx_out = y == 1

    n_f = x.shape[1]
    del_list = []
    for i in range(n_f):
        if np.std(x[:, i]) == 0:
            del_list.append(i)
    if len(del_list) > 0:
        logger.info("Pre-process: Delete %d features as every instances have the same behaviour: " % len(del_list))
        x = np.delete(x, del_list, axis=1)

    # keep outlier ratio, norm is normal out is outlier
    if seed == -1:
        rs = None
    else:
        rs = seed
    x_train_norm, x_test_norm, y_train_norm, y_test_norm = train_test_split(x[idx_norm], y[idx_norm],
                                                                            test_size=test_size,
                                                                            random_state=rs)
    x_train_out, x_test_out, y_train_out, y_test_out = train_test_split(x[idx_out], y[idx_out],
                                                                        test_size=test_size,
                                                                        random_state=rs)
    x_train = np.concatenate((x_train_norm, x_train_out))
    x_test = np.concatenate((x_test_norm, x_test_out))
    y_train = np.concatenate((y_train_norm, y_train_out))
    y_test = np.concatenate((y_test_norm, y_test_out))

    # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
    # scaler = StandardScaler().fit(x_train)
    # x_train = scaler.transform(x_train)
    # x_test = scaler.transform(x_test)

    # Scale to range [0,1]
    minmax_scaler = MinMaxScaler().fit(x_train)
    x_train = minmax_scaler.transform(x_train)
    x_test = minmax_scaler.transform(x_test)

    return x_train, y_train, x_test, y_test


def semi_setting(x_train, y_train, ratio_known_outliers, seed):
    outlier_indices = np.where(y_train == 1)[0]
    n_outliers = len(outlier_indices)

    if seed == -1:
        rng = np.random.RandomState(None)
    else:
        rng = np.random.RandomState(seed)

    n_known_outliers = int(round(n_outliers * ratio_known_outliers))
    known_idx = rng.choice(outlier_indices, n_known_outliers, replace=False)
    new_y_train = np.zeros(x_train.shape[0], dtype=int)
    new_y_train[known_idx] = 1
    return new_y_train


def get_sorted_index(score, order='descending'):
    '''
    :param score:
    :return: index of sorted item in descending order
    e.g. [8,3,4,9] return [3,0,2,1]
    '''
    score_map = []
    size = len(score)
    for i in range(size):
        score_map.append({'index':i, 'score':score[i]})
    if order == "descending":
        reverse = True
    elif order == "ascending":
        reverse = False
    score_map.sort(key=lambda x: x['score'], reverse=reverse)
    keys = [x['index'] for x in score_map]
    return keys


def get_rank(score):
    '''
    :param score:
    :return:
    e.g. input: [0.8, 0.4, 0.6] return [0, 2, 1]
    '''
    sort = np.argsort(score)
    size = score.shape[0]
    rank = np.zeros(size)
    for i in range(size):
        rank[sort[i]] = size - i - 1

    return rank


def min_max_norm(array):
    array = np.array(array)
    _min_, _max_ = np.min(array), np.max(array)
    if _min_ == _max_:
        raise ValueError("Given a array with same max and min value in normalisation")
    norm_array = np.array([(a - _min_) / (_max_ - _min_) for a in array])
    return norm_array


def sum_norm(array):
    array = np.array(array)
    sum = np.sum(array)
    norm_array = array / sum
    return norm_array


def get_performance(score, y_true):
    auc_roc = metrics.roc_auc_score(y_true, score)
    precision, recall, _ = metrics.precision_recall_curve(y_true, score)
    auc_pr = metrics.auc(recall, precision)
    return auc_roc, auc_pr


def ensemble_scores(score1, score2):
    '''
    :param score1:
    :param score2:
    :return: ensemble score
    @@ ensemble two score functions
    we use a non-parameter way to dynamically get the tradeoff between two estimated scores.
    It is much more important if one score function evaluate a object with high outlier socre,
    which should be paid more attention on these scoring results.
    instead of using simple average, median or other statistics
    '''

    objects_num = len(score1)

    [_max, _min] = [np.max(score1), np.min(score1)]
    score1 = (score1 - _min) / (_max - _min)
    [_max, _min] = [np.max(score2), np.min(score2)]
    score2 = (score2 - _min) / (_max - _min)

    rank1 = get_rank(score1)
    rank2 = get_rank(score2)

    alpha_list = (1. / (2 * (objects_num - 1))) * (rank2 - rank1) + 0.5
    combine_score = alpha_list * score1 + (1. - alpha_list) * score2
    return combine_score


def mat2csv(in_path, out_root_path):
    from scipy import io
    import pandas as pd
    data = io.loadmat(in_path)

    x = np.array(data['X'])
    y = np.array(data['y'], dtype=int)

    n_f = x.shape[1]
    columns = ["A" + str(i) for i in range(n_f)]
    columns.append("class")
    matrix = np.concatenate([x, y], axis=1)

    df = pd.DataFrame(matrix, columns=columns)

    name = in_path.split("/")[-1].split(".")[0]
    df.to_csv(out_root_path + name + ".csv", index=False)
    return


def get_summary(in_path):
    import pandas as pd
    name = in_path.split("/")[-1].split(".")[0]
    df = pd.read_csv(in_path)
    x = df.values[:, :-1]
    y = df.values[:, -1]
    n_x = x.shape[0]
    n_f = x.shape[1]
    n_ano = np.sum(y)
    print("%s, %d, %d, %d" % (name, n_x, n_f, n_ano))


def mnist_od(org_df, out_root_path, a):
    from numpy.random import RandomState

    x = org_df.values[:, :-1]
    y = org_df.values[:, -1]
    n_f = x.shape[1]

    if a == 1:
        # use one class as normal, and sampling anomalies from other classes, imbalance rate=1%
        for i in range(10):
            normal_ids = np.where(y == i)[0]
            n_normal = len(normal_ids)
            n_anomaly = int(n_normal * 0.01)

            for j in range(10):
                candidate_ids = np.where(y != i)[0]
                anomaly_ids = RandomState(None).choice(candidate_ids, n_anomaly, replace=False)

                normal_data = x[normal_ids]
                anomaly_data = x[anomaly_ids]
                n_all = n_normal + n_anomaly
                out_y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_anomaly, dtype=int)]).reshape([n_all, 1])
                out_x = np.concatenate([normal_data, anomaly_data], axis=0)
                print(out_x.shape, out_y.shape)
                matrix = np.concatenate([out_x, out_y], axis=1)
                columns = ["A" + str(i) for i in range(n_f)]
                columns.append("class")
                df = pd.DataFrame(matrix, columns=columns)
                df.to_csv(out_root_path + "mnist_" + str(i) + "-" + str(j) + ".csv", index=False)

    elif a == 2:
        # use one class as anomaly (100), and sampling inliers from other classes, imbalance rate=1%
        for i in range(10):
            for j in range(10):
                n_anomaly = 100
                anomaly_ids = RandomState(None).choice(np.where(y == i)[0], n_anomaly, replace=False)
                n_normal = 50 * n_anomaly
                normal_ids = RandomState(None).choice(np.where(y != i)[0], n_normal, replace=False)

                normal_data = x[normal_ids]
                anomaly_data = x[anomaly_ids]
                n_all = n_normal + n_anomaly
                out_y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_anomaly, dtype=int)]).reshape([n_all, 1])
                out_x = np.concatenate([normal_data, anomaly_data], axis=0)
                print(out_x.shape, out_y.shape)
                matrix = np.concatenate([out_x, out_y], axis=1)
                columns = ["A" + str(i) for i in range(n_f)]
                columns.append("class")
                df = pd.DataFrame(matrix, columns=columns)
                df.to_csv(out_root_path + "mnist2_A" + str(i) + "-" + str(j) + ".csv", index=False)

    elif a == 3:
        # use 0 as normal, 6 as anomaly
        for j in range(10):
            normal_ids = np.where(y == 0)[0]
            n_normal = len(normal_ids)
            n_anomaly = int(n_normal * 0.01)

            candidate_ids = np.where(y == 6)[0]
            anomaly_ids = RandomState(None).choice(candidate_ids, n_anomaly, replace=False)

            normal_data = x[normal_ids]
            anomaly_data = x[anomaly_ids]
            n_all = n_normal + n_anomaly
            out_y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_anomaly, dtype=int)]).reshape([n_all, 1])
            out_x = np.concatenate([normal_data, anomaly_data], axis=0)
            print(out_x.shape, out_y.shape)
            matrix = np.concatenate([out_x, out_y], axis=1)
            columns = ["A" + str(i) for i in range(n_f)]
            columns.append("class")
            df = pd.DataFrame(matrix, columns=columns)
            df.to_csv(out_root_path + "mnist2_A6N0" + "-" + str(j) + ".csv", index=False)


    return


if __name__ == '__main__':
    # input_root = "F:/OneDrive/work/data/ODDS/real/"
    input_root = "data/"
    import os
    if os.path.isdir(input_root):
        for file_name in sorted(os.listdir(input_root)):
            if file_name.endswith(".csv"):
                input_path = os.path.join(input_root, file_name)
                # mat2csv(input_path, "F:/data/DATASETS/ODDS/csv/")
                get_summary(input_path)

    # mnist_df = pd.read_csv("F:/OneDrive/work/data/mnist/org/all.csv")
    # mnist_od(mnist_df, out_root_path="F:/OneDrive/work/data/mnist/OD2new/", a=2)
