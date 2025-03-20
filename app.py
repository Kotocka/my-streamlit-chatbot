import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import networkx as nx
import matplotlib.pyplot as plt
import json
import os

# Файл для хранения истории чата
DATA_FILE = "chat_memory.json"

# Функция для загрузки/сохранения истории чата
def load_chat_history():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_chat_history(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

# Заголовок
st.title("🤖 Интерактивная диалоговая нейросеть")

# Функция загрузки модели GPT-2
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2", cache_dir="./cache")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", cache_dir="./cache")
    return tokenizer, model

tokenizer, model = load_model()

# **НАСТРОЙКИ НЕЙРОСЕТИ**
st.sidebar.header("⚙️ Настройки нейросети")
layers = st.sidebar.text_input("Слои (например, 10,20,10):", "10,20,10")
layers = list(map(int, layers.split(",")))

# Выбор функции активации (для визуализации)
activation = st.sidebar.selectbox("Функция активации:", ["relu", "sigmoid", "tanh"])

# Функция визуализации архитектуры сети
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
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=False, node_size=300, edge_color='gray')
    plt.title(f"🧠 Архитектура нейросети (Функция активации: {activation})")
    st.pyplot(plt)

# Кнопка "Обновить нейросеть"
if st.sidebar.button("Обновить нейросеть"):
    visualize_nn(layers)
    st.sidebar.write("✅ Нейросеть обновлена!")

# **ЧАТ-БОТ С ИСТОРИЕЙ**
st.subheader("💬 Введите ваш вопрос:")

# Загружаем историю чата
chat_history =
