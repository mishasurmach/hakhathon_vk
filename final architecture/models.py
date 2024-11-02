from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import catboost as cb
import lightgbm as lgb
from config import Config
import torch
import torch.nn as nn

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(
            C=Config.LOGISTIC_REGRESSION_C,
            solver=Config.LOGISTIC_REGRESSION_SOLVER,
            class_weight=Config.LOGISTIC_REGRESSION_CLASS_WEIGHT,
            max_iter=Config.LOGISTIC_REGRESSION_MAX_ITER
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=Config.RANDOM_FOREST_N_ESTIMATORS,
            max_depth=Config.RANDOM_FOREST_MAX_DEPTH,
            min_samples_split=Config.RANDOM_FOREST_MIN_SAMPLES_SPLIT,
            min_samples_leaf=Config.RANDOM_FOREST_MIN_SAMPLES_LEAF,
            class_weight=Config.RANDOM_FOREST_CLASS_WEIGHT,
            random_state=Config.RANDOM_FOREST_RANDOM_STATE,
            n_jobs=Config.RANDOM_FOREST_N_JOBS
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

class CatBoostRankerModel:
    def __init__(self):
        self.model = cb.CatBoostRanker(
            iterations=Config.CATBOOST_ITERATIONS,
            learning_rate=Config.CATBOOST_LEARNING_RATE,
            depth=Config.CATBOOST_DEPTH,
            verbose=Config.CATBOOST_VERBOSE
        )

    def train(self, X_train, y_train, group_train, X_val, y_val, group_val):
        train_pool = cb.Pool(data=X_train, label=y_train, group_id=group_train)
        val_pool = cb.Pool(data=X_val, label=y_val, group_id=group_val)
        self.model.fit(train_pool, eval_set=val_pool)

    def predict(self, X, group=None):
        pool = cb.Pool(data=X, group_id=group)
        return self.model.predict(pool)
    

class LightGBMRankerModel:
    def __init__(self):
        self.model = None
        self.params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'learning_rate': Config.LGB_LEARNING_RATE,
            'num_leaves': Config.LGB_NUM_LEAVES,
            'verbose': -1
        }

    def train(self, X_train, y_train, group_train, X_val, y_val, group_val):
        train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
        val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=Config.LGB_NUM_BOOST_ROUND,
        )

    def predict(self, X, group=None):
        return self.model.predict(X)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(772, Config.MLP_HIDDEN_DIMENSION_1)
        self.fc2 = nn.Linear(Config.MLP_HIDDEN_DIMENSION_1, Config.MLP_HIDDEN_DIMENSION_2)
        self.fc3 = nn.Linear(Config.MLP_HIDDEN_DIMENSION_2, 1)  

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def predict(self, X, group=None):
        return self.forward(torch.Tensor(X))
