import streamlit as st

import pandas as pd

product={}

product['pid'] = [101,102,103,104,105]
product['pname']=['pA', 'pB', 'pC', 'pD', 'pE']
product['pcost']=[1000,2000,3000,1500,2500]


df = pd.DataFrame(product)


st.write('Product Details:-')
st.write(df)


import numpy as np

data = pd.DataFrame(np.random.randn(20,5), columns=['F1','F2','F3','F4','F5'])
st.write(data)

st.line_chart(data)