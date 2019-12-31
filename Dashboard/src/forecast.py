import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from fbprophet import Prophet

import config


y_col = config.y_col
x_cols = config.x_cols
date_col = config.date_col
user_col = config.user_col
input_file = config.processed_data_file
prediction_result_file = config.prediction_result_file
prediction_error_file = config.prediction_error_file
n_test = config.n_test
n_thresh = config.n_thresh

df = pd.read_pickle(input_file)
df[date_col] = pd.to_datetime(df[date_col])
df.sort_values(date_col, inplace=True)

users = df[user_col].unique()
dates = df[date_col].unique().astype('datetime64[D]')

# forecast by Prophet
res_all = []
for user in tqdm(users):
    # filter by user ID
    d = df.loc[df[user_col] == user, [date_col, y_col] + x_cols]
    d.dropna(inplace=True)

    if len(d) < n_thresh:
        continue

    # split dataset
    n_train = len(d) - n_test

    d_pr = d.reset_index(drop=True)
    d_pr.columns = ['ds', 'y'] + x_cols

    d_pr_train, d_pr_test = d_pr.iloc[:n_train], d_pr.iloc[n_train:]

    # Python
    m = Prophet(yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=False)

    for x_col in x_cols:
        m.add_regressor(x_col, mode='additive')
    m.fit(d_pr_train)

    # Python
    future = d_pr.drop('y', axis=1)

    # Python
    pr_res = m.predict(future)

    # add true y and train-test label
    pr_res['y'] = d_pr['y']
    pr_res['group'] = 'train'
    pr_res.loc[n_train:, 'group'] = 'test'
    pr_res[user_col] = user

    res_all.append(pr_res)

res_all = pd.concat(res_all)
res_all.to_pickle(prediction_result_file)


# res_all = pd.read_pickle(prediction_result_file)
users = res_all[user_col].unique()

# 予測誤差算出
error_all = []
for user in tqdm(users):
    d = res_all.query(f'{user_col} == @user & group == "test"')

    square_error = np.square(d['y'] - d['yhat'])
    percentage_error = np.abs(d['y'] - d['yhat']) / d['y'] * 100

    user_error = pd.DataFrame({user_col: user,
                             'n_ahead': np.arange(n_test) + 1,
                             'se': square_error,
                             'pe': percentage_error})

    error_all.append(user_error)

error_all = pd.concat(error_all)
error_all.reset_index(drop=True, inplace=True)

error_all.to_pickle(prediction_error_file)
