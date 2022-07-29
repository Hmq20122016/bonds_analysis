import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
#streamlit run E:\Mengqi\Streamlit\Bonds.py
st.write("Bond Analysis")

df = pd.DataFrame(columns = ['Model', 'Type', 'Yield'])
df['Model'] = ['LGFV', 'CORP','LGFV', 'CORP','LGFV', 'CORP']
df['Type'] = ['1', '2','3', '3','4', '1']
df['Yield'] = [1,2,3,4,5,6]
st.sidebar.title("Please select the category")
Model = st.sidebar.multiselect('choose the model',list(df['Model'].unique())+['All'])
Type = st.sidebar.multiselect('choose the Type',list(df['Type'].unique())+['All'])


df_tmp = df.copy(deep = True)
if 'All' not in Model:
    df_tmp = df_tmp[df_tmp['Model'].isin(Model)]
if 'All' not in Type:
    df_tmp = df_tmp[df_tmp['Type'].isin(Type)]


st.dataframe(df_tmp) # will display the dataframe
fig = px.box(df_tmp['Yield'])
st.plotly_chart(fig)



        # state_total_graph = px.bar(
        #     state_total,
        #     x='病例分类',
        #     y='病例数',
        #     labels={'病例数': '%s 国家的总病例数' % (select)},
        #     color='病例分类')
        # st.plotly_chart(state_total_graph)
