import streamlit as st
import numpy as np
import joblib
from data_preprocesser import DataPreprocessor
from config import Config

# Функция загрузки модели
def load_model(filename):
    model = joblib.load(f"trained_models/{filename}")
    st.write(f"Модель загружена из {filename}.")
    return model

# Добавляем HTML и CSS для заголовка
st.markdown(
    """
    <h1 style='color: #0277ff;'>DreamTeam @ VK</h1>
    """, 
    unsafe_allow_html=True
)

# Ввод пути до модели
model_path = st.text_input("Введите название обученной модели:")

# Загрузка модели, если указали путь
if model_path:
    model = load_model(model_path)
else:
    st.warning("Пожалуйста, введите действительный путь до обученной модели.")

# Ввод query и passage
query = st.text_input("Введите запрос")
passage = st.text_area("Введите пассаж")

# Кнопка для проверки релевантности
if st.button("Проверить релевантность"):
    if model_path and query and passage:
        preprocessor = DataPreprocessor()
        datum = preprocessor.preprocess_single(query, passage)
        y_pred = model.predict(np.array(datum).reshape(1, -1))

        # Вывод результата
        if y_pred >= 0.5:
            st.write("Пассаж релевантен запросу.")
        else:
            st.write("Пассаж нерелевантен запросу.")
    else:
        st.warning("Please provide all inputs (model path, query, and passage).")
