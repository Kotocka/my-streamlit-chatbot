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
st.title("🤖 Умная диалоговая нейросеть")

# Функция загрузки модели DialoGPT
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", cache_dir="./cache")
    model = AutoModelFo
