import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import yfinance as yf
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("S&P 500 APP")
st.markdown("""
This app will show stocks in **S&P500** index
""")

st.sidebar.header("User Input Features")

@st.cache
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header=0)
    df =html[0]
    return df

df = load_data()
sector = df.groupby('GICS Sector')
sorted_sectors = sorted(df['GICS Sector'].unique())
select_sector = st.sidebar.multiselect("Sectors",sorted_sectors,sorted_sectors)
st.markdown('''
** Stocks: **
''')
df_sel_sector = df[df['GICS Sector'].isin(select_sector)]
st.write("Rows : "+str(df_sel_sector.shape[0])+" Column: "+str(df_sel_sector.shape[1]))
st.dataframe(df_sel_sector)

def download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href

st.markdown(download(df), unsafe_allow_html=True)
data = yf.download(
        tickers = list(df_sel_sector[:10].Symbol),
        period = "ytd",
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = True,
        proxy = None
    )
def price_plot(symbol):
  df = pd.DataFrame(data[symbol].Close)
  df['Date'] = df.index
  plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
  plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Closing Price', fontweight='bold')
  return st.pyplot()
num_company = st.sidebar.slider('Number of Companies', 1, 5)
if st.button('Show Plots'):
    st.header('Stock Closing Price')
    for i in list(df_sel_sector.Symbol)[:num_company]:
        price_plot(i)