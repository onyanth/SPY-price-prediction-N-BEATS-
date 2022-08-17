# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from tensorflow import keras
from plotly import graph_objs as go

# background information starts (first part)

df = pd.read_csv("C:/Users/user/Desktop/SPX_5min.csv").drop(['Open', 'High', 'Low'],axis=1) 
df["Close"] = round(df["Close"])

new_model = keras.models.load_model('C:/Users/user/Desktop/model_7.name')

def plot_raw_data():
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=df["DateTime"], y=df["Close"][-390:], name="Closing price"))
  fig.layout.update(title_text = 'Time Series Data (nearest 5 days)', xaxis_rangeslider_visible=True)
  st.plotly_chart(fig)

# background information ends (first part)

# Sidebar content starts
st.sidebar.markdown('Each timestep represents one 5-min interval.')
n_periods = st.sidebar.selectbox('How many timesteps do you want to predict?',[0,1,2,3,4,5,6,7])


p1 = p2 = p3 = p4 = p5 = p6 = int()
if n_periods == 2 :
  p1 = st.sidebar.number_input('Enter the lastest closed price')
elif n_periods == 3 :
  p2 = st.sidebar.number_input('Enter the secound lastest closed price')
  p1 = st.sidebar.number_input('Enter the lastest closed price')  
elif n_periods == 4 :
  p3 = st.sidebar.number_input('Enter the third lastest closed price')
  p2 = st.sidebar.number_input('Enter the secound lastest closed price') 
  p1 = st.sidebar.number_input('Enter the lastest closed price')
elif n_periods == 5 :
  p4 = st.sidebar.number_input('Enter the fourth lastest closed price') 
  p3 = st.sidebar.number_input('Enter the third lastest closed price')  
  p2 = st.sidebar.number_input('Enter the secound lastest closed price')  
  p1 = st.sidebar.number_input('Enter the lastest closed price')  
elif n_periods == 6 :
  p5 = st.sidebar.number_input('Enter the fifth lastest closed price')
  p4 = st.sidebar.number_input('Enter the fourth lastest closed price')  
  p3 = st.sidebar.number_input('Enter the third lastest closed price')  
  p2 = st.sidebar.number_input('Enter the secound lastest closed price')  
  p1 = st.sidebar.number_input('Enter the lastest closed price')  
elif n_periods == 7 :
  p6 = st.sidebar.number_input('Enter the sixth lastest closed price')
  p5 = st.sidebar.number_input('Enter the fifth lastest closed price')  
  p4 = st.sidebar.number_input('Enter the fourth lastest closed price')  
  p3 = st.sidebar.number_input('Enter the third lastest closed price')  
  p2 = st.sidebar.number_input('Enter the secound lastest closed price')  
  p1 = st.sidebar.number_input('Enter the lastest closed price')  

# Sidebar content ends

# Main page content starts

st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
st.title("SPY Closing Price prediction")

plot_raw_data()

st.subheader('Closing Price')


# Main page content ends 

#backend operation

if n_periods != 0:
    if n_periods == 1 :
        element = st.dataframe(df.iloc[-78:])
    elif n_periods == 2 :
        element = st.dataframe(df.iloc[-78:])
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:00'],'Close':[float(p1)]}))
    elif n_periods == 3 :
        element = st.dataframe(df.iloc[-78:])
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:00'],'Close':[float(p2)]}))
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:05'],'Close':[float(p1)]}))
    elif n_periods == 4 :
        element = st.dataframe(df.iloc[-78:])
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:00'],'Close':[float(p3)]}))
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:05'],'Close':[float(p2)]}))
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:10'],'Close':[float(p1)]}))
    elif n_periods == 5 :
        element = st.dataframe(df.iloc[-78:])
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:00'],'Close':[float(p4)]}))
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:05'],'Close':[float(p3)]}))
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:10'],'Close':[float(p2)]}))
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:15'],'Close':[float(p1)]}))
    elif n_periods == 6 :
        element = st.dataframe(df.iloc[-78:])
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:00'],'Close':[float(p5)]}))
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:05'],'Close':[float(p4)]}))
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:10'],'Close':[float(p3)]}))
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:15'],'Close':[float(p2)]}))
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:20'],'Close':[float(p1)]}))
    elif n_periods == 7 :
        element = st.dataframe(df.iloc[-78:])
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:00'],'Close':[float(p6)]}))
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:05'],'Close':[float(p5)]}))
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:10'],'Close':[float(p4)]}))
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:15'],'Close':[float(p3)]}))
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:20'],'Close':[float(p2)]}))
        element.add_rows(pd.DataFrame({'DataTime':['7/2/2022 09:25'],'Close':[float(p1)]}))
    else:
        pass  

df_def = df.copy()

def format_fitting(last = 1):
    data_of_prediction = pd.DataFrame()
    for i in range(7):
      data_of_prediction[f"Close+{i+1}"] = df_def["Close"].shift(periods=i+1)
    data_of_prediction = data_of_prediction.iloc[296637:,:]
    return (pd.DataFrame(data_of_prediction.iloc[-last]).transpose())

for i in range(7):
  df[f"Price+{i+1}"] = df["Close"].shift(periods=i+1)

df.drop(["DateTime","Close"], axis=1, inplace=True)


pre1 = new_model.predict(format_fitting(last = 1))
r_s_1 = df.iloc[-1:].shift(periods= 1, axis = 1)
r_s_1.at[296644,"Price+1"]=pre1[0][0].round(decimals=1)

pre2 = new_model.predict(r_s_1)
r_s_2 = r_s_1.shift(periods= 1, axis = 1)
r_s_2.at[296644,"Price+1"]=pre2[0][0].round(decimals=1)

pre3 = new_model.predict(r_s_2)
r_s_3 = r_s_2.shift(periods= 1, axis = 1)
r_s_3.at[296644,"Price+1"]=pre3[0][0].round(decimals=1)

pre4 = new_model.predict(r_s_3)
r_s_4 = r_s_3.shift(periods= 1, axis = 1)
r_s_4.at[296644,"Price+1"]=pre4[0][0].round(decimals=1)

pre5 = new_model.predict(r_s_4)
r_s_5 = r_s_4.shift(periods= 1, axis = 1)
r_s_5.at[296644,"Price+1"]=pre5[0][0].round(decimals=1)

pre6 = new_model.predict(r_s_5)
r_s_6 = r_s_5.shift(periods= 1, axis = 1)
r_s_6.at[296644,"Price+1"]=pre6[0][0].round(decimals=1)

pre7 = new_model.predict(r_s_6).round(decimals=1)



if n_periods == 1 :
    st.dataframe(format_fitting(last = 1))
    st.metric(label="Predicted Closing Price", value= pre1[0][0].round(decimals=1))
elif n_periods == 2 :
    if p1!=0:
        first = df.iloc[-1:].shift(periods= 1, axis = 1)
        first.at[296644,"Price+1"]=p1
        st.metric(label="Predicted Closing Price", value= pre2[0][0].round(decimals=1))
        st.metric(label="Adjusted Predicted Closing Price", value= new_model.predict(first))
    else:
        st.dataframe(r_s_1.round(decimals = 1))
        st.metric(label="Predicted Closing Price", value= pre2[0][0].round(decimals=1))
elif n_periods == 3 :
    if p2!=0 and p1==0:
        secound = df.iloc[-1:].shift(periods= 1, axis = 1)
        secound.at[296644,"Price+1"]=p2
        st.dataframe(secound)
        st.metric(label="Predicted Closing Price", value= pre3[0][0].round(decimals=1))
        st.metric(label="Adjusted Predicted Closing Price", value= new_model.predict(secound))
    elif p1!=0 and p2!=0:
        secound = df.iloc[-1:].shift(periods= 1, axis = 1)
        secound.at[296644,"Price+1"]=p2 
        first = secound.shift(periods= 1, axis = 1)
        first.at[296644,"Price+1"]=p1
        st.dataframe(first)
        st.metric(label="Predicted Closing Price", value= pre3[0][0].round(decimals=1))
        st.metric(label="Adjusted Predicted Closing Price", value= new_model.predict(first))
    else:
        st.dataframe(r_s_2.round(decimals = 1))
        st.metric(label="Predicted Closing Price", value= pre3[0][0].round(decimals=1))
elif n_periods == 4 :
    st.dataframe(r_s_3.round(decimals = 1))
    st.metric(label="Predicted Closing Price", value= pre4[0][0].round(decimals=1))
elif n_periods == 5 :
    st.dataframe(r_s_4.round(decimals = 1))
    st.metric(label="Predicted Closing Price", value= pre5[0][0].round(decimals=1))
elif n_periods == 6 :
    st.dataframe(r_s_5.round(decimals = 1))
    st.metric(label="Predicted Closing Price", value= pre6[0][0].round(decimals=1))
elif n_periods == 7 :
        st.dataframe(r_s_6.round(decimals = 1))
        st.metric(label="Predicted Closing Price", value= pre7[0][0].round(decimals=1))
else:
    pass 




