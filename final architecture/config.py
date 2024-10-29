import time

class Config:
    # Пути к данным
    TRAIN_DATA_PATH = 'data/train_data.pkl'
    VAL_DATA_PATH = 'data/val_data.pkl'
    TEST_DATA_PATH = 'data/test_data.pkl'
    
    # Гиперпараметры для логистической регрессии
    LOGISTIC_REGRESSION_C = 1.0
    LOGISTIC_REGRESSION_SOLVER = 'lbfgs'
    LOGISTIC_REGRESSION_CLASS_WEIGHT = 'balanced'
    LOGISTIC_REGRESSION_MAX_ITER = 10000
    
    # Гиперпараметры для случайного леса
    RANDOM_FOREST_N_ESTIMATORS = 100
    RANDOM_FOREST_MAX_DEPTH = None
    RANDOM_FOREST_MIN_SAMPLES_SPLIT = 2
    RANDOM_FOREST_MIN_SAMPLES_LEAF = 1
    RANDOM_FOREST_CLASS_WEIGHT = 'balanced'
    RANDOM_FOREST_RANDOM_STATE = 42
    RANDOM_FOREST_N_JOBS = -1
    
    # Гиперпараметры для CatBoost
    CATBOOST_ITERATIONS = 100
    CATBOOST_LEARNING_RATE = 0.1
    CATBOOST_DEPTH = 6
    CATBOOST_VERBOSE = 100
    
    # Гиперпараметры для DataLoader и модели
    MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    ADDITIONAL_EMB_COLUMNS = ['euclidean_dist', 'manhattan_dist', 'dot_product', 'chebyshev_dist']
    
    # Параметры для расчета метрик
    #METRIC_NDCG_GROUP_SIZE = 10  # Пример, если нужно учитывать размер групп
    
    # Логирование и визуализация
    ENABLE_LOGGING = True
    ROC_CURVE_PLOT_PATH = f'{time.time()}_roc_curve.png'  # Путь для сохранения графика ROC-кривой
