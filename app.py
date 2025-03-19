import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Функция для визуализации структуры нейросети
def visualize_nn(layers):
    G = nx.DiGraph()
    for i, neurons in enumerate(layers):
        for j in range(neurons):
            G.add_node(f"L{i}_N{j}", layer=i)
        if i > 0:
            for prev in range(layers[i-1]):
                for curr in range(neurons):
                    G.add_edge(f"L{i-1}_N{prev}", f"L{i}_N{curr}")

    pos = nx.multipartite_layout(G, subset_key="layer")
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=False, node_size=300, edge_color='gray')
    plt.title("Структура нейросети")
    st.pyplot(plt)

# Создание модели
def create_model(layer_sizes, activation):
    model = keras.Sequential()
    model.add(keras.layers.Dense(layer_sizes[0], activation=activation, input_shape=(10,)))  # Входной слой
    for size in layer_sizes[1:]:
        model.add(keras.layers.Dense(size, activation=activation))  # Скрытые слои
    model.add(keras.layers.Dense(1, activation="sigmoid"))  # Выходной слой
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model

st.title("Диалоговая нейросеть")

# Настройки нейросети
st.sidebar.header("Настройки нейросети")

# Ввод количества слоев
layers = st.sidebar.text_input("Введите слои (например, 10,20,10):", "10,20,10")
layers = list(map(int, layers.split(",")))

# Выбор функции активации
activation = st.sidebar.selectbox("Функция активации", ["relu", "sigmoid", "tanh"])

# Количество эпох
epochs = st.sidebar.slider("Количество эпох", 1, 20, 5)

# Кнопка для обновления модели
if st.sidebar.button("Обновить нейросеть"):
    model = create_model(layers, activation)
    visualize_nn(layers)
    st.sidebar.write("✅ Нейросеть обновлена!")

# Поле ввода текста
question = st.text_input("Введите ваш вопрос:")

# Кнопка отправки
if st.button("Отправить"):
    if "model" not in locals():
        model = create_model(layers, activation)  # Создаём модель, если её нет
    input_data = np.random.rand(1, 10)  # Генерируем случайный вектор
    response = model.predict(input_data)[0][0]  # Прогоняем через нейросеть
    st.write(f"Ответ: {response:.4f}")
