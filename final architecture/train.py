import pandas as pd
import numpy as np
from config import Config
from models import LogisticRegressionModel, RandomForestModel, CatBoostRankerModel, LightGBMRankerModel
import catboost as cb
from collections import Counter
from evaluate import evaluate
import joblib
import time


# import onnx
# import onnxmltools 
# from skl2onnx import convert_sklearn
# from skl2onnx.common.data_types import FloatTensorType

def load_data():
    df_train = pd.read_pickle(Config.TRAIN_DATA_PATH)
    df_val = pd.read_pickle(Config.VAL_DATA_PATH)
    df_test = pd.read_pickle(Config.TEST_DATA_PATH)
    return df_train, df_val, df_test

def prepare_data(df, GBM_model = False, query_emb_col='query_emb', passage_emb_col='passage_emb', label_col='label', query_id_col='query_id'):
    X = np.hstack((np.stack(df[query_emb_col].values), np.stack(df[passage_emb_col].values)))
    
    for col in Config.ADDITIONAL_EMB_COLUMNS:
        additional_data = np.stack(df[col].values).reshape(-1, 1)
        X = np.hstack((X, additional_data))
    
    y = df[label_col].values
    if GBM_model == False:
        group = df[query_id_col].values
    else:
        query_counts = Counter(df[query_id_col])
        group = [query_counts[query_id] for query_id in query_counts]

    return X, y, group

# def save_model_onnx(model, X_sample, model_name="model.onnx"):
#     # Задаем начальные типы данных на основе данных X_sample
#     n_features = X_sample.shape[1]
#     initial_type = [("input", FloatTensorType([None, n_features]))]
#     onnx_model = convert_sklearn(model, initial_types=initial_type)
#     onnxmltools.utils.save_model(onnx_model, model_name)

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def train_logistic_regression(df_train, df_test):
    X_train, y_train, group_train = prepare_data(df_train)
    X_test, y_test, group_test = prepare_data(df_test)

    model = LogisticRegressionModel()
    model.train(X_train, y_train)
    
    path = f'trained_models/log_reg_model_{time.time()}.pkl'
    save_model(model, path)

    y_pred = model.predict(X_test)
    return y_pred, y_test, group_test

def train_random_forest(df_train, df_test):
    X_train, y_train, group_train = prepare_data(df_train)
    X_test, y_test, group_test = prepare_data(df_test)

    model = RandomForestModel()
    model.train(X_train, y_train)
    
    # Save model
    path = f'trained_models/random_forest_model_{time.time()}.pkl'
    save_model(model, path)
    
    y_pred = model.predict(X_test)
    return y_pred, y_test, group_test

def train_catboost(df_train, df_test, df_val):
    X_train, y_train, group_train = prepare_data(df_train)
    X_val, y_val, group_val = prepare_data(df_val)
    X_test, y_test, group_test = prepare_data(df_test)

    model = CatBoostRankerModel()
    model.train(X_train, y_train, group_train, X_val, y_val, group_val)
    
    # Save model
    path = f'trained_models/catboost_model_{time.time()}.pkl'
    save_model(model, path)
    
    y_pred = model.predict(X_test, group_test)
    return y_pred, y_test, group_test

def train_lightGBM(df_train, df_test, df_val):
    X_train, y_train, group_train = prepare_data(df_train, GBM_model = True)
    X_val, y_val, group_val = prepare_data(df_val, GBM_model = True)
    X_test, y_test, group_test = prepare_data(df_test, GBM_model = True)

    model = LightGBMRankerModel()
    model.train(X_train, y_train, group_train, X_val, y_val, group_val)
    
    # Save model
    path = f'trained_models/lightGBM_model_{time.time()}.pkl'
    save_model(model, path)
    
    y_pred = model.predict(X_test, group_test)
    return y_pred, y_test, group_test


if __name__ == "__main__":
    df_train, df_val, df_test = load_data()

    if (Config.MODEL_TO_TRAIN == 'LogReg'):
        y_pred, y_test, group_test = train_logistic_regression(df_train, df_test)
        evaluate(y_pred, y_test, group_test)

    if (Config.MODEL_TO_TRAIN == 'CatBoost'):
        y_pred, y_test, group_test = train_catboost(df_train, df_test, df_val)
        evaluate(y_pred, y_test, group_test)

    if (Config.MODEL_TO_TRAIN == 'Tree'):
        y_pred, y_test, group_test = train_random_forest(df_train, df_test)
        evaluate(y_pred, y_test, group_test)

    if (Config.MODEL_TO_TRAIN == 'LightGBM'):
        y_pred, y_test, group_test = train_lightGBM(df_train, df_test, df_val)
        evaluate(y_pred, y_test, group_test)