import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

st.title("🤖 Улучшенная диалоговая нейросеть")

# Кэшируем модель, чтобы не загружать её при каждом запуске
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2", cache_dir="./cache")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", cache_dir="./cache")
    return tokenizer, model

tokenizer, model = load_model()

# Поле ввода
question = st.text_input("Введите ваш вопрос:")

if st.button("Отправить"):
    inputs = tokenizer.encode(question, return_tensors="pt")
    response_ids = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    
    st.write(f"🤖 **Ответ:** {response}")
