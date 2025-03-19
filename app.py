import streamlit as st

st.title("Диалоговая нейросеть")

question = st.text_input("Введите ваш вопрос:")

if st.button("Отправить"):
    st.write("Ответ: Привет, я нейросеть!")

st.sidebar.header("Настройки")

layers = st.sidebar.text_input("Введите слои (например, 10,20,10):", "10,20,10")
activation = st.sidebar.selectbox("Функция активации", ["relu", "sigmoid", "tanh"])
epochs = st.sidebar.slider("Эпохи", 1, 20, 5)

st.sidebar.write(f"Функция активации: {activation}")
st.sidebar.write(f"Эпохи: {epochs}")
