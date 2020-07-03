import numpy as np
import pandas as pd


def generate_data2(n_nor, dim, rate=0.01, max_nc=1, max_ac=1):
    n_ano = int(n_nor * rate)

    normal_loc = np.zeros([dim, max_nc])
    normal_scale = np.zeros([dim, max_nc])
    anomaly_loc = np.zeros([dim, max_ac])
    anomaly_scale = np.zeros([dim, max_ac])

    for i in range(dim):
        for j in range(max_nc):
            normal_loc[i][j] = np.random.rand()
            normal_scale[i][j] = np.random.rand()
        for j in range(max_ac):
            if np.random.randint(0, 1):
                anomaly_loc[i][j] = np.random.rand() + 0.6
            else:
                anomaly_loc[i][j] = np.random.rand() - 0.6

            anomaly_scale[i][j] = np.random.rand()

    for ac in range(1, max_ac+1):
        for nc in range(1, max_nc+1):
            print("ac", ac, "nc", nc)
            # normal class with "n_nor_c" clusters
            x_nor = np.zeros([n_nor, dim])
            for d in range(dim):
                size = round(n_nor / nc)
                for c in range(nc):
                    loc = normal_loc[d][c]
                    scale = normal_loc[d][c]
                    # print("Inlier: dim"+str(d), "cluster"+str(c), round(loc, 1), round(scale, 2))
                    # last c
                    if c == nc - 1:
                        last_size = n_nor - (nc-1)*size
                        x_nor[c * size:, d] = np.random.normal(loc, scale, last_size)
                    else:
                        x_nor[c * size: (c+1)*size, d] = np.random.normal(loc, scale, size)

            x_ano = np.zeros([n_ano, dim])
            for d in range(dim):
                size = round(n_ano / ac)
                for c in range(ac):
                    loc = anomaly_loc[d][c]
                    scale = anomaly_scale[d][c]
                    # print("anomaly: dim"+str(d), "cluster"+str(c), round(loc, 1), round(scale, 2))

                    if c != ac - 1:
                        x_ano[c*size: (c+1)*size, d] = np.random.normal(loc, scale, size)
                    else:
                        last_size = n_ano - (ac - 1) * size
                        x_ano[c*size:, d] = np.random.normal(loc, scale, last_size)

            x = np.concatenate([x_ano, x_nor], axis=0)
            y = np.append(np.ones(n_ano, dtype=int), np.zeros(n_nor, dtype=int))
            matrix = np.concatenate([x, y.reshape([x.shape[0], 1])], axis=1)

            columns = ["A"+str(i) for i in range(dim)]
            columns.append("class")
            df = pd.DataFrame(matrix, columns=columns)
            df['class'] = df['class'].astype(int)
            out_path = "data/synthetic2/synnew_" + "a" + str(ac) + "n" + str(nc) + ".csv"
            df.to_csv(out_path, index=False)
    return


def generate_data(n_nor, n_ano, dim, rate=0, n_nor_c=1, n_ano_c=1):
    if rate > 0:
        n_ano = int(n_nor * rate)

    # normal class with "n_nor_c" clusters
    x_nor = np.zeros([n_nor, dim])
    for i in range(dim):
        size = round(n_nor / n_nor_c)
        for j in range(n_nor_c):
            loc = np.random.rand()
            scale = float(np.random.rand())
            print("Inlier: dim"+str(i), "cluster"+str(j), round(loc, 1), round(scale, 2))
            # last c
            if j == n_nor_c - 1:
                last_size = n_nor - (n_nor_c-1)*size
                x_nor[j * size:, i] = np.random.normal(loc, scale, last_size)
            else:
                x_nor[j * size: (j+1)*size, i] = np.random.normal(loc, scale, size)

    x_ano = np.zeros([n_ano, dim])
    for i in range(dim):
        size = round(n_ano / n_ano_c)
        for j in range(n_ano_c):
            loc = np.random.rand() + 1
            scale = float(np.random.rand())
            print("anomaly: dim"+str(i), "cluster"+str(j), round(loc, 1), round(scale, 2))

            # last c
            if j != n_ano_c - 1:
                x_ano[j*size: (j+1)*size, i] = np.random.normal(loc, scale, size)
            else:
                last_size = n_ano - (n_ano_c - 1) * size
                x_ano[j*size:, i] = np.random.normal(loc, scale, last_size)
            # x_ano[:, i] = np.random.normal(loc, scale, n_ano)

    x = np.concatenate([x_ano, x_nor], axis=0)
    y = np.append(np.ones(n_ano, dtype=int), np.zeros(n_nor, dtype=int))
    matrix = np.concatenate([x, y.reshape([x.shape[0], 1])], axis=1)

    columns = ["A"+str(i) for i in range(dim)]
    columns.append("class")
    df = pd.DataFrame(matrix, columns=columns)
    df['class'] = df['class'].astype(int)
    return df


def generate_data_scal_dim(n_nor, n_ano, dim_list, rate=0, n_nor_c=1, n_ano_c=1):
    if rate > 0:
        n_ano = int(n_nor * rate)

    dim_max = max(dim_list)
    # normal class with "n_nor_c" clusters
    x_nor = np.zeros([n_nor, dim_max])
    for i in range(dim_max):
        size = round(n_nor / n_nor_c)
        for j in range(n_nor_c):
            loc = np.random.rand()
            scale = float(np.random.rand())
            print("Inlier: dim"+str(i), "cluster"+str(j), round(loc, 1), round(scale, 2))
            # last c
            if j == n_nor_c - 1:
                last_size = n_nor - (n_nor_c-1)*size
                x_nor[j * size:, i] = np.random.normal(loc, scale, last_size)
            else:
                x_nor[j * size: (j+1)*size, i] = np.random.normal(loc, scale, size)

    x_ano = np.zeros([n_ano, dim_max])
    for i in range(dim_max):
        size = round(n_ano / n_ano_c)
        for j in range(n_ano_c):
            loc = np.random.rand() + 1
            scale = float(np.random.rand())
            print("anomaly: dim"+str(i), "cluster"+str(j), round(loc, 1), round(scale, 2))

            # last c
            if j != n_ano_c - 1:
                x_ano[j*size: (j+1)*size, i] = np.random.normal(loc, scale, size)
            else:
                last_size = n_ano - (n_ano_c - 1) * size
                x_ano[j*size:, i] = np.random.normal(loc, scale, last_size)
            # x_ano[:, i] = np.random.normal(loc, scale, n_ano)

    x = np.concatenate([x_ano, x_nor], axis=0)
    y = np.append(np.ones(n_ano, dtype=int), np.zeros(n_nor, dtype=int))

    df_lst = []
    for dim in dim_list:
        x_subset = x[:, :dim]
        matrix = np.concatenate([x_subset, y.reshape([x.shape[0], 1])], axis=1)

        columns = ["A"+str(i) for i in range(dim)]
        columns.append("class")
        df = pd.DataFrame(matrix, columns=columns)
        df['class'] = df['class'].astype(int)
        df_lst.append(df)

    return df_lst


# for i in range(10):
#     for j in range(10):
#         df = generate_data(n_nor=5000, dim=10, rate=0.01, n_nor_c=i+1, n_ano_c=j+1)
#         out_path = "data/synthetic/syn_" + "a" + str(j+1) + "n" + str(i+1) + ".csv"
#         print(out_path)
#         df.to_csv(out_path, index=False)

# TEST the significance of surrogate supervision-based deviation learning
generate_data2(n_nor=1000, dim=30, rate=0.05, max_nc=10, max_ac=10)


# # Scal-up Test
# dim_range = [32, 64, 128, 256, 512, 1024, 2048]
# size_range = [2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]
#
#
# root = "data/scal_test/dim/"
# for ii, dim in enumerate(dim_range):
#     n_nor = 950
#     n_ano = 50
#     size = n_nor + n_ano
#     df = generate_data(n_nor=n_nor, n_ano=n_ano, dim=dim)
#     name = "scal_dim" + str(ii) + "_" + str(size) + "-" + str(dim) + ".csv"
#     df.to_csv(root + name, index=False)

# for ii, size in enumerate(size_range):
#     dim = 32
#     n_nor = int(size * 0.95)
#     n_ano = int(size * 0.05)
#     df = generate_data(n_nor=n_nor, n_ano=n_ano, dim=dim)
#     name = "scal_size" + str(ii) + "_" + str(size) + "-" + str(dim) + ".csv"
#     df.to_csv(root + name, index=False)

# df_lst = generate_data_scal_dim(n_nor=1900, n_ano=100, dim_list=dim_range)
# for ii, df in enumerate(df_lst):
#     size = 2000
#     dim = dim_range[ii]
#     name = "scal_dim" + str(ii) + "_" + str(size) + "-" + str(dim) + ".csv"
#     df.to_csv(root + name, index=False)
