import time

class Config:
    PYTHON = 'python3.11'

    # модель для обучения
    # MODEL_TO_TRAIN = 'LogReg'
    # MODEL_TO_TRAIN = 'CatBoost'
    # MODEL_TO_TRAIN = 'Tree'
    # MODEL_TO_TRAIN = 'LightGBM'
    MODEL_TO_TRAIN = 'MLP'

    MODEL_TO_INFERENCE_PATH = 'trained_models/catboost_model_1730246243.224709.pkl'

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

    # Гиперпараметры для LightGBM
    LGB_NUM_BOOST_ROUND = 100
    LGB_LEARNING_RATE  = 0.2
    LGB_NUM_LEAVES = 127

    # Гиперпараметры для MLP
    MLP_HIDDEN_DIMENSION_1 = 128
    MLP_HIDDEN_DIMENSION_2 = 50
    MLP_LEARNING_RATE = 0.003
    MLP_NUMBER_OF_EPOCHS = 50
    
    # Гиперпараметры для DataLoader и модели
    MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    ADDITIONAL_EMB_COLUMNS = []#['euclidean_dist', 'manhattan_dist', 'dot_product', 'chebyshev_dist']
    
    # Логирование и визуализация
    ENABLE_LOGGING = True
    ROC_CURVE_PLOT_PATH = f'graphs/{time.time()}_roc_curve.pdf'  # Путь для сохранения графика ROC-кривой
