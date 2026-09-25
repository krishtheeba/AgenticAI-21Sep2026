import streamlit as st

st.title("Welcome")

st.header("Header Line")

st.write("Sample data")

st.text("Sample data")

name= st.text_input("Enter your name")

st.text(f'Hello ..... {name}')



model = st.text_input("Enter a model name")


if model :
	st.text(f'Model Name is :{model}')
