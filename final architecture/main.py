import subprocess
from config import Config

def run_data_preprocessor():
    print("Запуск скачивания и предобработки данных...")
    subprocess.run([Config.PYTHON, "data_preprocesser.py"])
    print("Скачивание и предобработка данных завершены. Данные сохранены в путь, указанный в config.py.")

def run_train():
    print("Запуск обучения модели...")
    subprocess.run([Config.PYTHON, "train.py"])
    print("Обучение завершено.")

if __name__ == "__main__":
    run_data_preprocessor()
    run_train()