import pandas as pd
import numpy as np
from config import Config
from models import LogisticRegressionModel, RandomForestModel, CatBoostRankerModel
from evaluate import evaluate

def load_data():
    df_train = pd.read_pickle(Config.TRAIN_DATA_PATH)
    df_val = pd.read_pickle(Config.VAL_DATA_PATH)
    df_test = pd.read_pickle(Config.TEST_DATA_PATH)
    return df_train, df_val, df_test

def prepare_data(df, query_emb_col='query_emb', passage_emb_col='passage_emb', label_col='label', query_id_col='query_id'):
    X = np.hstack((np.stack(df[query_emb_col].values), np.stack(df[passage_emb_col].values)))
    
    for col in Config.ADDITIONAL_EMB_COLUMNS:
        additional_data = np.stack(df[col].values).reshape(-1, 1)
        X = np.hstack((X, additional_data))
    
    y = df[label_col].values
    group = df[query_id_col].values
    return X, y, group

def train_logistic_regression(df_train, df_test):
    X_train, y_train, group_train = prepare_data(df_train)
    X_test, y_test, group_test = prepare_data(df_test)

    model = LogisticRegressionModel()
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, y_test, group_test

def train_random_forest(df_train, df_test):
    X_train, y_train, group_train = prepare_data(df_train)
    X_test, y_test, group_test = prepare_data(df_test)

    model = RandomForestModel()
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, y_test, group_test

def train_catboost(df_train, df_test, df_val):
    X_train, y_train, group_train = prepare_data(df_train)
    X_val, y_val, group_val = prepare_data(df_val)
    X_test, y_test, group_test = prepare_data(df_test)

    model = CatBoostRankerModel()
    model.train(X_train, y_train, group_train, X_val, y_val, group_val)
    y_pred = model.predict(X_test, group_test)
    return y_pred, y_test, group_test

if __name__ == "__main__":
    df_train, df_val, df_test = load_data()

    # Обучение и оценка логистической регрессии
    y_pred, y_test, group_test = train_logistic_regression(df_train, df_test)
    evaluate(y_pred, y_test, group_test)

    # # Обучение и оценка случайного леса
    # y_pred, y_test, group_test = train_random_forest(df_train, df_test)
    # evaluate(y_pred, y_test, group_test)

    # # Обучение и оценка CatBoost
    # y_pred, y_test, group_test = train_catboost(df_train, df_test, df_val)
    # evaluate(y_pred, y_test, group_test)

