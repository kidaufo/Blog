y_col = 'from_count'
x_cols = ['Mean_Temperature_F', 'MeanDew_Point_F']
date_col ='Date'
user_col = 'station_id'
user_type_col = 'station_type_1'
n_test = 100 # number of test data
n_thresh = 200 # minimum number of observation
processed_data_file = '/Users/yuho.kida/ARISE/需要予測アセット/data/trip_station_weather.pkl'
prediction_result_file = '/Users/yuho.kida/ARISE/需要予測アセット/data/prophet_res.pkl'
prediction_error_file = '/Users/yuho.kida/ARISE/需要予測アセット/data/prediction_error.pkl'
