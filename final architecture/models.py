from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# import catboost as cb
from config import Config

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
