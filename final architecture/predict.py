import pandas as pd
from config import Config
from models import LogisticRegressionModel, RandomForestModel, CatBoostRankerModel
from data_preprocesser import DataPreprocessor

class Predictor:
    def __init__(self, model_type='logistic_regression'):
        self.preprocessor = DataPreprocessor()
        
        # Загрузка нужной модели в зависимости от типа
        if model_type == 'logistic_regression':
            self.model = LogisticRegressionModel()
        elif model_type == 'random_forest':
            self.model = RandomForestModel()
        elif model_type == 'catboost':
            self.model = CatBoostRankerModel()
        else:
            raise ValueError("Unsupported model type. Choose from 'logistic_regression', 'random_forest', 'catboost'.")
        
        # Загрузка обученной модели
        self.load_model()

    def load_model(self):
        # Загружаем обученную модель из файла или директории, если это требуется.
        # Пример:
        # self.model.load('path/to/model')
        pass  # здесь заглушка, реализуйте загрузку, если это необходимо

    def predict_single(self, query, passage_text):
        # Предобработка запроса и текста пассажа
        processed_data = self.preprocessor.preprocess_single(query, passage_text)
        
        # Подготовка данных для предсказания
        features = self._prepare_features(processed_data)
        
        # Выполнение предсказания
        prediction = self.model.predict([features])
        
        # Возвращаем результат и рассчитанные меры схожести
        return {
            'prediction': prediction[0],
            'cosine_sim': processed_data['cosine_sim'],
            'euclidean_dist': processed_data['euclidean_dist'],
            'manhattan_dist': processed_data['manhattan_dist'],
            'dot_product': processed_data['dot_product'],
            'chebyshev_dist': processed_data['chebyshev_dist']
        }

    def _prepare_features(self, processed_data):
        # Подготовка данных для модели: создание вектора признаков
        query_emb = processed_data['query_emb']
        passage_emb = processed_data['passage_emb']
        
        # Создаем вектор признаков, объединяя эмбеддинги и меры схожести
        features = list(query_emb) + list(passage_emb)
        for col in Config.ADDITIONAL_EMB_COLUMNS:
            features.append(processed_data[col])
        
        return features

# Пример использования
if __name__ == "__main__":
    predictor = Predictor(model_type='logistic_regression')  # или другой тип модели
    query = "Что такое машинное обучение?"
    passage_text = "Машинное обучение - это область искусственного интеллекта, изучающая методы автоматического построения алгоритмов."
    
    result = predictor.predict_single(query, passage_text)
    print("Предсказание:", result['prediction'])
    print("Косинусное сходство:", result['cosine_sim'])
    print("Евклидово расстояние:", result['euclidean_dist'])
    print("Манхэттенское расстояние:", result['manhattan_dist'])
    print("Скалярное произведение:", result['dot_product'])
    print("Чебышёвское расстояние:", result['chebyshev_dist'])
