from tcn import compiled_tcn
from utils import get_xy_kfolds
from sklearn.metrics import mean_squared_error
import numpy as np

# dataset source: https://github.com/laiguokun/multivariate-time-series-data
# exchange rate: the collection of the daily exchange rates of eight foreign countries
# including Australia, British, Canada, Switzerland, China, Japan, New Zealand and
# Singapore ranging from 1990 to 2016.
# task: predict multi-column daily exchange rate from history

folds, enc = get_xy_kfolds()
mse_list = []

if __name__ == '__main__':
    mse_list = []
    for train_x, train_y, test_x, test_y in folds:
        print("sfsdf")
        print(len(train_x[0]))
        print(type(train_y))
        print(train_y[:,0])
        print(test_x.shape[1])
        print(train_x.shape[2])
        print(test_x.shape[0])
        print(train_x.shape[1])
        print(test_y.shape[0])
        print(train_y.shape[1])
        # exit()

        model = compiled_tcn(return_sequences=False,
                             num_feat=train_x.shape[2],
                             nb_filters=24,
                             num_classes=0,
                             kernel_size=8,
                             dilations=[2 ** i for i in range(9)],
                             nb_stacks=1,
                             max_len=train_x.shape[1],
                             use_skip_connections=True,
                             regression=True,
                             dropout_rate=0,
                             output_len=train_y.shape[1])
        model.fit(train_x, train_y[:,0], batch_size=256, epochs=20)
        y_raw_pred = model.predict(np.array([test_x]))

        y_true = enc.inverse_transform([test_y]).flatten()
        print(y_raw_pred)
        print(y_true)
        exit()
        y_pred = enc.inverse_transform(y_raw_pred).flatten()
        y_true = enc.inverse_transform([test_y]).flatten()
        mse_cur = mean_squared_error(y_true, y_pred)
        mse_list.append(mse_cur)
        print(f"train_set_size:{train_x.shape[0]}")
        print(f"y_true:{y_true}")
        print(f"y_pred:{y_pred}")
        print(f"mse:{mse_cur}")
    print(f"finial loss on test set: {np.mean(mse_list)}")
