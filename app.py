import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import networkx as nx
import matplotlib.pyplot as plt

# ЗАГОЛОВОК
st.title("🤖 Интерактивная нейросеть для диалогов")

# ФУНКЦИЯ ЗАГРУЗКИ МОДЕЛИ GPT-2
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2", cache_dir="./cache")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", cache_dir="./cache")
    return tokenizer, model

tokenizer, model = load_model()

# 🛠️ **НАСТРОЙКИ НЕЙРОСЕТИ**
st.sidebar.header("⚙️ Настройки нейросети")
layers = st.sidebar.text_input("Слои (например, 10,20,10):", "10,20,10")
layers = list(map(int, layers.split(",")))
activation = st.sidebar.selectbox("Функция активации:", ["relu", "sigmoid", "tanh"])

# ФУНКЦИЯ ВИЗУАЛИЗАЦИИ АРХИТЕКТУРЫ НЕЙРОСЕТИ
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
    plt.title("🧠 Структура нейросети")
    st.pyplot(plt)

# КНОПКА "ОБНОВИТЬ НЕЙРОСЕТЬ"
if st.sidebar.button("Обновить нейросеть"):
    visualize_nn(layers)
    st.sidebar.write("✅ Нейросеть обновлена!")

# 📝 **ЧАТ-БОТ**
st.subheader("💬 Введите ваш вопрос:")
question = st.text_input("")

if st.button("Отправить"):
    inputs = tokenizer.encode(question, return_tensors="pt")
    response_ids = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)

    st.write(f"🤖 **Ответ:** {response}")

# ВИЗУАЛИЗАЦИЯ СТРУКТУРЫ НЕЙРОСЕТИ
st.subheader("🕸️ Архитектура нейросети:")
visualize_nn(layers)
