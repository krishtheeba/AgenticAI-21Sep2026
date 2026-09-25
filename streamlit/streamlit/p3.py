import streamlit as st
import pandas as pd

st.set_page_config(page_title="LB Dashboard", layout="centered")

st.title("Welcome to Streamlit")

st.write("Hello... user A")


embedding_model = st.text_input("Enter the Embedding model Name:")

if embedding_model:
	st.write(f"Input Model is {embedding_model}")


llm_model = st.text_input("Enter the LLM model Name:")

if llm_model:
	st.success(f"Input Model is {llm_model}")



vector = st.slider("Select the vector range")
if vector:
	st.write(f'selected vector range:{vector}')

vector = st.slider("Select the vector range",500, 1000)
if vector:
	st.write(f'selected vector range:{vector}')


vector = st.slider("Select the vector range",5000,6000,5560)
if vector:
	st.write(f'selected vector range:{vector}')



st.select_slider("Select the rate:", ["Bad","Good","Excellent"])


file = st.file_uploader("select your input file:")


if file:
	st.success(f'Input file :{file.name} is loaded')
	df=pd.read_csv(file)
	st.write(df)


st.checkbox("yes")
st.checkbox("no")

st.radio("select your interface:", ['eth0','eth1','eth2'])

age = st.number_input("Enter your Age:",5,90)

st.write(age)

st.date_input('Travel date')



















