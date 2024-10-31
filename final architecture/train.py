import pandas as pd
import numpy as np
from config import Config
from models import LogisticRegressionModel, RandomForestModel, CatBoostRankerModel, LightGBMRankerModel, MLP
import catboost as cb
from collections import Counter
from evaluate import evaluate
from evaluate import calculate_ndcg_by_group
from sklearn.metrics import average_precision_score, roc_auc_score
import joblib
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim

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

def train_iterative(model, X_train, y_train, X_test, y_test, group_test, lr=0.001, epochs = 20, number_of_exp = 1):
    writer = SummaryWriter(log_dir=f"runs/experiment{number_of_exp}")
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Логирование метрик каждые 10 эпох
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                y_pred = (model(X_test).squeeze() > 0.5).float()
                print(y_pred)
                precision = average_precision_score(y_test.numpy(), y_pred.numpy())
                ndcg = calculate_ndcg_by_group(y_test.numpy(), y_pred.numpy(), group_test)
                auc = roc_auc_score(y_test.numpy(), y_pred.numpy())

                writer.add_scalar("Loss", loss.item(), epoch)
                writer.add_scalar("Average Precision", precision, epoch)
                writer.add_scalar("NDCG", ndcg, epoch)
                writer.add_scalar("AUC", auc, epoch)
    
    writer.close()

def train_MLP(df_train, df_test):
    X_train, y_train, group_train = prepare_data(df_train)
    X_test, y_test, group_test = prepare_data(df_test)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
        
    model = MLP()
    train_iterative(model, X_train, y_train, X_test, y_test, group_test, 
                             Config.MLP_LEARNING_RATE, Config.MLP_NUMBER_OF_EPOCHS)
    
    path = f'trained_models/MLP_{time.time()}.pkl'
    save_model(model, path)

    with torch.no_grad():
        y_pred = (model(X_test).squeeze() > 0.5).float()

    return y_pred.numpy(), y_test.numpy(), group_test

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

    if (Config.MODEL_TO_TRAIN == 'MLP'):
        y_pred, y_test, group_test = train_MLP(df_train, df_test)
        evaluate(y_pred, y_test, group_test)

