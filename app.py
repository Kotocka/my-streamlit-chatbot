import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import networkx as nx
import matplotlib.pyplot as plt
import json
import os

# Файл, где хранятся данные для обучения (память чата)
DATA_FILE = "chat_memory.json"

# Функция для загрузки/сохранения данных чата
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

# Функция для визуализации структуры нейросети
def visualize_nn(model):
    G = nx.DiGraph()

    layers = []
    for i, (name, param) in enumerate(model.named_parameters()):
        if "weight" in name:
            layers.append(param.shape[0])  # Количество нейронов в слое

    # Создание узлов и связей между слоями
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

# Загружаем предобученную модель GPT-2
@st.cache_resource
def load_gpt2():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_gpt2()

st.title("🤖 Диалоговая нейросеть с интерактивной визуализацией")

# Загружаем историю чата
chat_history = load_data()

# Поле ввода текста
question = st.text_input("Введите ваш вопрос:")

# Кнопка отправки
if st.button("Отправить"):
    # Проверяем, есть ли этот вопрос в истории
    existing_answer = next((entry["answer"] for entry in chat_history if entry["question"] == question), None)

    if existing_answer:
        response = existing_answer  # Если вопрос уже был, используем старый ответ
    else:
        # Генерируем новый ответ с помощью GPT-2
        inputs = tokenizer.encode(question + tokenizer.eos_token, return_tensors="pt")
        response_ids = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(response_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)

        # Запоминаем новый ответ
        chat_history.append({"question": question, "answer": response})
        save_data(chat_history)

    st.write(f"🧠 **Ответ:** {response}")

# Выводим историю диалога
st.subheader("📝 История диалога:")
for entry in chat_history[-5:]:  # Показываем последние 5 сообщений
    st.write(f"**Q:** {entry['question']}")
    st.write(f"**A:** {entry['answer']}")
    st.write("---")

# Визуализация структуры нейросети
st.subheader("🕸️ Визуализация нейросети:")
visualize_nn(model)
