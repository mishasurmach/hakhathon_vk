import pandas as pd
import numpy as np
from datasets import load_dataset
from scipy.spatial import distance
from config import Config
from sentence_transformers import SentenceTransformer

class DataPreprocessor:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        self.dataset_train = None
        self.dataset_dev = None

    def load_datasets(self):
        self.dataset_train = load_dataset("cohere/miracl-ru-queries-22-12", split="train")
        self.dataset_dev = load_dataset("cohere/miracl-ru-queries-22-12", split="dev")


    def split_datasets(self):
        train_df = self.dataset_train.to_pandas().drop(columns=['emb'])
        dev_df = self.dataset_dev.to_pandas().drop(columns=['emb'])
        
        df_train = train_df[:4155]
        df_val = pd.concat([train_df[4155:-200], dev_df[:362], dev_df[-200:]])
        df_test = pd.concat([train_df[-200:], dev_df[362:-200]])
        
        return df_train, df_val, df_test

    def encode_queries(self, df, column_name):
        return self.model.encode(df[column_name].tolist())

    def process_passages(self, row):
        new_rows = []
        query_emb = row['query_emb']
        
        # Positive passages
        for pos_passage in row['positive_passages']:
            new_rows.append({
                'query_emb': query_emb,
                'query_id': row['query_id'],
                'passage_text': pos_passage['text'], 
                'label': 1
            })

        # Negative passages
        for neg_passage in row['negative_passages']:
            new_rows.append({
                'query_emb': query_emb,
                'query_id': row['query_id'],
                'passage_text': neg_passage['text'],
                'label': 0
            })
            
        return new_rows

    def create_expanded_dataset(self, df):
        new_rows = []
        for _, row in df.iterrows():
            new_rows.extend(self.process_passages(row))
        return pd.DataFrame(new_rows)

    def encode_passages(self, df, column_name):
        return self.model.encode(df[column_name].tolist())

    def calculate_similarity_measures(self, query_emb, passage_emb):
        return {
            'cosine_sim': 1 - distance.cosine(query_emb, passage_emb),
            'euclidean_dist': distance.euclidean(query_emb, passage_emb),
            'manhattan_dist': distance.cityblock(query_emb, passage_emb),
            'dot_product': np.dot(query_emb, passage_emb),
            'chebyshev_dist': distance.chebyshev(query_emb, passage_emb)
        }

    def add_similarity_measures(self, df):
        df['similarity_measures'] = df.apply(
            lambda row: self.calculate_similarity_measures(row['query_emb'], row['passage_emb']), axis=1
        )
        return pd.concat([df, df['similarity_measures'].apply(pd.Series)], axis=1).drop(columns=['similarity_measures'])

    def preprocess_and_save(self):
        # Разделение данных
        df_train, df_val, df_test = self.split_datasets()
        
        # Кодирование запросов
        df_train['query_emb'] = self.encode_queries(df_train, 'query')
        df_val['query_emb'] = self.encode_queries(df_val, 'query')
        df_test['query_emb'] = self.encode_queries(df_test, 'query')
        
        # Создание расширенного набора данных с позитивными и негативными примерами
        df_train = self.create_expanded_dataset(df_train)
        df_val = self.create_expanded_dataset(df_val)
        df_test = self.create_expanded_dataset(df_test)
        
        # Кодирование текстов пассажа
        df_train['passage_emb'] = self.encode_passages(df_train, 'passage_text')
        df_val['passage_emb'] = self.encode_passages(df_val, 'passage_text')
        df_test['passage_emb'] = self.encode_passages(df_test, 'passage_text')
        
        # Добавление мер близости
        df_train = self.add_similarity_measures(df_train).drop(columns=['passage_text'])
        df_val = self.add_similarity_measures(df_val).drop(columns=['passage_text'])
        df_test = self.add_similarity_measures(df_test).drop(columns=['passage_text'])
        
        # Сохранение данных
        df_train.to_pickle(Config.TRAIN_DATA_PATH)
        df_val.to_pickle(Config.VAL_DATA_PATH)
        df_test.to_pickle(Config.TEST_DATA_PATH)

    def preprocess_single(self, query, passage_text):
        # Кодирование запроса и текста пассажа
        query_emb = self.model.encode([query])[0]
        passage_emb = self.model.encode([passage_text])[0]

        # Вычисление мер близости
        similarity_measures = self.calculate_similarity_measures(query_emb, passage_emb)

        output = np.hstack((query_emb, passage_emb))
        for col in Config.ADDITIONAL_EMB_COLUMNS:
            output = np.hstack((output,similarity_measures[col]))
            
        return output
    
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.load_datasets()  # Загрузка данных при необходимости
    preprocessor.preprocess_and_save()
