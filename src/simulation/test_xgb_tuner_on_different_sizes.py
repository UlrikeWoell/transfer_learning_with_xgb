from src.xgb_tuning.xgb import XGBTunerRand
from src.util.data_in_out import read_data


data = read_data('size_10000.csv')
feature_cols = [col for col in data if col.startswith('X')]
target_col = 'transformed_target'

x_features = data[feature_cols]
y_target = data[target_col]
tuner = XGBTunerRand()
tuner.tune(x_features=x_features, y_target=y_target)