import torch
import torch.utils.data as Data
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
import utils
from log import logger

from sklearn.metrics import confusion_matrix


def train(x, y, net, optimizer, loss_func, device, batch_size=64, n_epochs=50, verbose=True):
    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.int64).to(device)
    train_dataset = Data.TensorDataset(x, y)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    for epoch in tqdm(range(n_epochs)):
        epoch_loss = 0.0
        n_batches: int = 0
        epoch_start_time = time.time()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            # out = net(batch_x)
            out = net(batch_x)[0]
            loss = loss_func(out, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            n_batches += 1
            if step == 1000:
                break

        epoch_train_time = time.time() - epoch_start_time
        if (epoch+1) % 5 == 0:
            if verbose:
                logger.debug(f'| Epoch: {epoch + 1:03}/{n_epochs:03} | Train Time: {epoch_train_time:.3f}s'
                             f'| Train Loss: {epoch_loss / n_batches:.6f} | Batch Size({batch_size})')
    return net


def calc_efficacy(x, y, net, device, ano_size, nor_size, verbose=False):
    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.int64).to(device)

    with torch.no_grad():
        # out = F.softmax(net(x), dim=1).cpu()
        out = F.softmax(net(x)[0], dim=1).cpu()
        _, y_pred = torch.max(out.data, 1)
        y_prob = out.cpu().data.numpy()

    if verbose:
        y = y.cpu().data.numpy()
        y_pred = y_pred.cpu().data.numpy()
        evaluate(y_pred, y)

    efficacy = np.zeros(nor_size)
    for i in range(ano_size):
        for j in range(nor_size):
            index = i * nor_size + j
            this_score = y_prob[index]
            efficacy[j] += this_score[1] - this_score[0]
    efficacy = efficacy / ano_size
    return efficacy


def predict(x, y, net, device, verbose=False):
    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.int64).to(device)

    with torch.no_grad():
        out = F.softmax(net(x), dim=1).cpu()
        _, y_pred = torch.max(out.data, 1)
        y_prob = out.data.numpy()

    y = y.cpu().data.numpy()
    y_pred = y_pred.cpu().data.numpy()
    if verbose:
        evaluate(y_pred, y)

    return y_pred, y_prob


def pure_predict(x, net, device):
    x = torch.tensor(x, dtype=torch.float32).to(device)

    with torch.no_grad():
        # out = F.softmax(net(x), dim=1).cpu()
        out = F.softmax(net(x)[0], dim=1).cpu()
        _, y_pred = torch.max(out.data, 1)
        y_prob = out.data.numpy()
    y_pred = y_pred.numpy()
    return y_pred, y_prob


def pure_predict_full_out(x, net, device):
    x = torch.tensor(x, dtype=torch.float32).to(device)

    with torch.no_grad():
        net_out, out_dic = net(x)
        out = F.softmax(net_out, dim=1).cpu()
        _, y_pred = torch.max(out.data, 1)
        y_prob = out.data.numpy()
    y_pred = y_pred.numpy()
    return y_pred, y_prob, out_dic


def test(x, y, net, batch_size=64):
    n_class = len(np.unique(y))
    x_test = torch.tensor(x, dtype=torch.float32)
    y_test = torch.tensor(y, dtype=torch.int64)
    test_dataset = Data.TensorDataset(x_test, y_test)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    correct = 0
    correct_list = np.zeros(n_class)
    predict = []
    ground_truth = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            output = net(batch_x)
            _, batch_predict = torch.max(output.data, 1)
            correct += (batch_predict == batch_y).sum().item()

            predict.extend(batch_predict.numpy().tolist())
            ground_truth.extend(batch_y.numpy().tolist())

            mark = (batch_predict == batch_y).numpy()
            for i in range(n_class):
                correct_list[i] += len(np.intersect1d(np.where(mark == 1)[0], np.where(batch_y == i)[0]))

    predict = np.array(predict)

    class_total = np.zeros(n_class)
    predict_total = np.zeros(n_class)
    for i in range(n_class):
        class_total[i] = len(np.where(y == i)[0])
        predict_total[i] = len(np.where(predict == i)[0])

    for i in range(n_class):
        print("Ground Truth: class%d: %d" % (i, class_total[i]))
    for i in range(n_class):
        print("Predict: class%d: %d" % (i, predict_total[i]))

    print('Total Acc: %.2f %%' % (100 * correct / len(y)))
    for i in range(n_class):
        print("Class %d Acc, %.2f%%" % (i,  100 * correct_list[i] / class_total[i]))


def evaluate(y_pred, y):
    """
    :param y_pred: tensor type
    :param y: ground truth y (tensor type)
    :return: ensemble score
    """
    n_y = y.shape[0]
    correct_list = np.zeros(2, dtype=int)
    mark = y_pred == y
    for i in range(2):
        correct_list[i] += len(np.intersect1d(np.where(mark == 1)[0], np.where(y == i)[0]))
    correct = correct_list.sum()
    class_total, predict_total = np.zeros(2, dtype=int), np.zeros(2, dtype=int)
    for i in range(2):
        class_total[i] = len(np.where(y == i)[0])
        predict_total[i] = len(np.where(y_pred == i)[0])

    print("class num:  ", end="")
    for i in range(2):
        print("class%d (true/pred): %d/%d, " % (i, class_total[i], predict_total[i]), end="")
    print()
    print('Total ACC:  %.2f%% -- ' % (100 * correct / n_y), end="")
    for i in range(2):
        print("class%d %.2f%%, " % (i, 100 * correct_list[i] / class_total[i]), end="")
    print()

    # cm = confusion_matrix(y.numpy(), y_pred.numpy())
    # print(cm)
    # ----------------------#
