import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from plotly import graph_objs as go
#GUI
st.markdown("<h1 style='text-align: center; color: Black; position:relative;top:-40px;'> Stock Predictor </h1>",unsafe_allow_html=True )
st.sidebar.markdown("<h1 style='text-align: left; color: Black; position:relative;top:-20px;'> Options </h1>",unsafe_allow_html=True)
stock = st.sidebar.selectbox("Select Stock", ("AAPL", "AMZN", "TSLA", "NVDA", "MSFT", "NFLX", "AAL", "DIS", "GOOG", "META"))
shares = st.sidebar.number_input("Number of Shares", min_value = 1)
period = st.sidebar.number_input("Number of Months to Predict", min_value = 1,max_value = 12)
st.sidebar.markdown("<h1 style='text-align: left; color: Black; position:relative;top:-10px;'> Compare </h1>",unsafe_allow_html=True)
stock2 = st.sidebar.selectbox("Select 2nd Stock to Compare", ("None","AAPL", "AMZN", "TSLA", "NVDA", "MSFT", "NFLX", "AAL", "DIS", "GOOG", "META"))


#Data collection/formatting
data = yf.download(stock, start = '2021-01-01', end = str(date.today()))
data.reset_index(inplace=True)
table = data[['Date','Close']]
table = data.rename(columns={"Date": "ds", "Close":"y"})
if stock2 != "None":
    data2 = yf.download(stock2, start = '2021-01-01', end = str(date.today()))
    data2.reset_index(inplace=True)
    table2 = data2[['Date','Close']]
    table2 = data2.rename(columns={"Date": "ds", "Close":"y"})
    model1 = Prophet()
    model1.fit(table2)
    frame1 = model1.make_future_dataframe(periods=period*30)
    forecast1 = model1.predict(frame1)
    print(forecast1)          
#Fitting data into model and making predictions
model = Prophet()
model.fit(table)
frame = model.make_future_dataframe(periods=period*30)
forecast = model.predict(frame)
graph = go.Figure()
graph.update_layout(title_text=f"{stock} Stock Price Prediction")
graph.update_layout(xaxis_title="Date", yaxis_title="Stock Price")
graph.add_trace(go.Scatter(x=forecast['ds'], y = forecast['yhat'], name = stock, mode ='lines'))
if stock2!="None":
    graph.add_trace(go.Scatter(x=forecast1['ds'], y = forecast1['yhat'], name = stock2, mode = 'lines'))
    graph.update_layout(title_text=f"{stock} vs {stock2} Comparison")
st.plotly_chart(graph)

#Predicting profit
dates = []
prices = []
profits = []
initial = yf.Ticker(stock).info['currentPrice']
table = forecast[['ds','yhat']]
for row in table.itertuples():
    if str(row.ds)[8:10] == '01' and int(str(row.ds)[0:4]) > 2023:
        dates.append(str(row.ds)[:10])
        prices.append(round(row.yhat,2))
        profits.append(round(float(row.yhat-initial)*shares,2))
if stock2 == 'None':
    ptable = go.Figure(data=[go.Table(header=dict(values=['Date', 'Price', 'Profit']), cells = dict(values=[dates, prices, profits]))])
    ptable.update_layout(title_text=f"{stock} Stock Profit Prediction")
    st.plotly_chart(ptable)
else: 
    prices1 = []
    profits1 = []
    initial1 = yf.Ticker(stock2).info['currentPrice']
    table1 = forecast1[['ds','yhat']]
    for row in table1.itertuples():
        if str(row.ds)[8:10] == '01' and int(str(row.ds)[0:4]) > 2023:
            prices1.append(round(row.yhat,2))
            profits1.append(round(float(row.yhat-initial1)*shares,2))
    ptable = go.Figure(data=[go.Table(header=dict(values=['Date', 'Price 1', 'Price 2', 'Profit 1', 'Profit 2']), cells = dict(values=[dates, prices, prices1, profits, profits1]))])
    ptable.update_layout(title_text="Prediction Comparison")
    st.plotly_chart(ptable)