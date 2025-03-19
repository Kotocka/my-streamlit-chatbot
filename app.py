import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import os

# Файл, где хранятся данные для обучения
DATA_FILE = "chat_memory.json"

# Функция для загрузки/сохранения данных
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

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
    model.compile(opti
