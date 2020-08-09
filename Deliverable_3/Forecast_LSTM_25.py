
# coding: utf-8

# In[56]:


#Import all the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import seaborn as sns
import math
import os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from IPython.display import display_html, HTML
import time
from sklearn.metrics import mean_squared_error
import random as rn
import tensorflow as tf
from pandas import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, kpss
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from pmdarima.arima.utils import ndiffs
from pandas.plotting import autocorrelation_plot
import plotly.graph_objects as go
import plotly.io as pio


# In[57]:


#put random seed
sd=1
np.random.seed(sd)
rn.seed(sd)
os.environ['PYTHONHASHSEED'] = str(sd)


# In[58]:


#set environment for consistency

from keras import backend as K

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.random.set_seed(sd)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config = config)
tf.compat.v1.keras.backend.set_session(sess)


# In[59]:


#Load the global confirmed cases by date datasets
case = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df_case = pd.read_csv(case)
df_case.head()


# In[60]:


#Drop a Province/State, Lat, Long columns, combine rows with same country name, change Country/Region to Country, and reset index

df_case = df_case.drop(df_case.columns[[0, 2, 3]], axis=1)
df_case = df_case.groupby(df_case['Country/Region']).aggregate('sum')
df_case.index.names = ['Country']
df_case.reset_index(level=0, inplace=True)
df_case.head()


# In[61]:


#Melt the dates to convert from columns to rows

column_list = df_case.columns.tolist()
date_columns = column_list[1:-1]

df_melt = df_case.melt(id_vars =['Country'], value_vars = date_columns,  
        var_name ='Date', value_name ='Total_case')
df_melt['Date'] = pd.to_datetime(df_melt['Date'])
df_melt['Date'] = df_melt['Date'].dt.strftime('%m/%d/%Y')
df_melt = df_melt.sort_values(["Date", "Country"])
df_melt = df_melt.reset_index()
df_melt = df_melt.drop('index', 1)
df_melt.head(-10)


# In[62]:


#chech df_melt shape

df_melt.shape


# In[63]:


#Change few country names in the dataframe

df_melt = df_melt.replace({'Country':{'United Kingdom':'UK',
                                      'South Africa':'South_Africa', 
                                      'Korea, South':'South_Korea',
                                      'New Zealand':'New_Zealand', 
                                      'Taiwan*':'Taiwan'}})


# In[64]:


#Group by country among 25 countries

country_25 = ['US','Brazil','Russia','India','UK','Peru','Chile',
              'Spain','Italy','Iran','France','Germany', 'Turkey', 
              'Mexico','South_Africa','China','Sweden','Singapore', 
              'South_Korea', 'Slovenia','New_Zealand','Vietnam',
              'Taiwan','Cambodia','Laos']


# In[65]:


#make dataframe with only countries in country_25 to reduce loop time

df = df_melt.loc[df_melt.Country.isin(country_25)]
df


# In[92]:


#loop to create dataframe for each country in country_25

gbl = globals()

for i in df.Country:
    if i in country_25:
        gbl['df_'+i] = df.loc[df.Country == i]
        gbl['df_'+i] = gbl['df_'+i].drop('Country', axis=1)
        gbl['df_'+i] = gbl['df_'+i].set_index('Date')
        gbl['df_'+i].index = pd.to_datetime(gbl['df_'+i].index, format='%m/%d/%Y'). strftime('%Y-%m-%d')


# In[93]:


#loop preprocessing and generate timeseries for 25 countries

from keras.preprocessing.sequence import TimeseriesGenerator

gbl = globals()
scaler = MinMaxScaler(feature_range=(0, 1))

for i in df.Country:
    if i in country_25:
        gbl['df_'+i+ '_scaled']= scaler.fit_transform(gbl['df_'+i])


# In[94]:


#n_input used US dataset since all dataset lengths are same
from keras.preprocessing.sequence import TimeseriesGenerator

gbl = globals()


for i in df.Country:
    if i in country_25:
        n_input = 90
        n_features = 1
        gbl['generator_' + i] = TimeseriesGenerator(gbl['df_'+i+ '_scaled'],gbl['df_'+i+ '_scaled'], 
                                n_input, batch_size=30)
        


# In[95]:


#build keras lstm model

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

lstm_model = Sequential()
lstm_model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_input, n_features)))
lstm_model.add(LSTM(80, activation='relu', return_sequences=True))
lstm_model.add(LSTM(50, activation='relu'))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.summary()


# In[63]:


#US


# In[133]:


train_US = df_US

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_US)
train_US_scaled = scaler.transform(train_US)


n_input = 90
n_features = 1
generator_US = TimeseriesGenerator(train_US_scaled,train_US_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_US, steps_per_epoch=1, epochs=100,
                        verbose=1)


# In[134]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_US.png');

#predict with lstm

US_list = list()
batch = train_US_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    US_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[US_list[i]]],axis=1)


# In[135]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_US.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_US_predict = pd.DataFrame(scaler.inverse_transform(US_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_US_predict= pd.concat([future_date, df_US_predict], axis=1)
df_US_predict= df_US_predict.set_index('Date')
df_US_proj = pd.concat([df_US, df_US_predict], axis=1)


df_US_proj.index=pd.to_datetime(df_US_proj.index)


# In[136]:


#plot numbers as log scale

fig_US = go.Figure()
fig_US.add_trace(go.Scatter(x=df_US_proj.index, 
                        y=np.log10(df_US_proj.Total_case+1),
                        mode='lines',
                        name='Actual'))
fig_US.add_trace(go.Scatter(x=df_US_proj.index, 
                        y=np.log10(df_US_proj.Prediction),
                        mode='lines',
                        name='Prediction'))

fig_US.update_layout(title='Log scale of COVID-19 Cases Prediction in US')
fig_US.show()


# In[137]:


#save the graph

pio.write_html(fig_US, file=r'/home/songy4/Documents/fig_US.html', auto_open=True)


# In[ ]:


#Brazil


# In[138]:


#preprocess, timeseries generator, 

train_Brazil = df_Brazil

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_Brazil)
train_Brazil_scaled = scaler.transform(train_Brazil)


n_input = 90
n_features = 1
generator_Brazil = TimeseriesGenerator(train_Brazil_scaled,train_Brazil_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_Brazil, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

Brazil_list = list()
batch = train_Brazil_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Brazil_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Brazil_list[i]]],axis=1)


# In[139]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_Brazil.png');

#predict with lstm

Brazil_list = list()
batch = train_Brazil_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Brazil_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Brazil_list[i]]],axis=1)


# In[140]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_Brazil.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_Brazil_predict = pd.DataFrame(scaler.inverse_transform(Brazil_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_Brazil_predict= pd.concat([future_date, df_Brazil_predict], axis=1)
df_Brazil_predict= df_Brazil_predict.set_index('Date')
df_Brazil_proj = pd.concat([df_Brazil, df_Brazil_predict], axis=1)


df_Brazil_proj.index=pd.to_datetime(df_Brazil_proj.index)


# In[176]:


#plot numbers as log scale

fig_Brazil = go.Figure()
fig_Brazil.add_trace(go.Scatter(x=df_Brazil_proj.index, 
                        y=np.log10(df_Brazil_proj.Total_case+1),
                        mode='lines',
                        name='Actual'))
fig_Brazil.add_trace(go.Scatter(x=df_Brazil_proj.index, 
                        y=np.log10(df_Brazil_proj.Prediction),
                        mode='lines',
                        name='Prediction'))

fig_Brazil.update_layout(title='Log scale of COVID-19 Cases Prediction in Brazil')
fig_Brazil.show()


# In[142]:


#save the graph

pio.write_html(fig_Brazil, file=r'/home/songy4/Documents/fig_Brazil.html', auto_open=True)


# In[ ]:


#Russia


# In[143]:


#preprocess, timeseries generator

train_Russia = df_Russia

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_Russia)
train_Russia_scaled = scaler.transform(train_Russia)


n_input = 90
n_features = 1
generator_Russia = TimeseriesGenerator(train_Russia_scaled,train_Russia_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_Russia, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

Russia_list = list()
batch = train_Russia_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Russia_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Russia_list[i]]],axis=1)


# In[144]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_Russia.png');

#predict with lstm

Russia_list = list()
batch = train_Russia_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Russia_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Russia_list[i]]],axis=1)


# In[145]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_Russia.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_Russia_predict = pd.DataFrame(scaler.inverse_transform(Russia_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_Russia_predict= pd.concat([future_date, df_Russia_predict], axis=1)
df_Russia_predict= df_Russia_predict.set_index('Date')
df_Russia_proj = pd.concat([df_Russia, df_Russia_predict], axis=1)


df_Russia_proj.index=pd.to_datetime(df_Russia_proj.index)


# In[173]:


#plot numbers as log scale

fig_Russia = go.Figure()
fig_Russia.add_trace(go.Scatter(x=df_Russia_proj.index, 
                        y=df_Russia_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_Russia.add_trace(go.Scatter(x=df_Russia_proj.index, 
                        y=df_Russia_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_Russia.update_layout(title='COVID-19 Cases Prediction in Russia')
fig_Russia.show()


# In[174]:


#save the graph

pio.write_html(fig_Russia, file=r'/home/songy4/Documents/fig_Russia.html', auto_open=True)


# In[ ]:


#India


# In[148]:


#preprocess, timeseries generator

train_India = df_India

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_India)
train_India_scaled = scaler.transform(train_India)


n_input = 90
n_features = 1
generator_India = TimeseriesGenerator(train_India_scaled,train_India_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_India, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

India_list = list()
batch = train_India_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    India_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[India_list[i]]],axis=1)


# In[149]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_India.png');

#predict with lstm

India_list = list()
batch = train_India_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    India_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[India_list[i]]],axis=1)


# In[150]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_India.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_India_predict = pd.DataFrame(scaler.inverse_transform(India_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_India_predict= pd.concat([future_date, df_India_predict], axis=1)
df_India_predict= df_India_predict.set_index('Date')
df_India_proj = pd.concat([df_India, df_India_predict], axis=1)


df_India_proj.index=pd.to_datetime(df_India_proj.index)


# In[170]:


#plot numbers as log scale

fig_India = go.Figure()
fig_India.add_trace(go.Scatter(x=df_India_proj.index, 
                        y=np.log10(df_India_proj.Total_case+1),
                        mode='lines',
                        name='Actual'))
fig_India.add_trace(go.Scatter(x=df_India_proj.index, 
                        y=np.log10(df_India_proj.Prediction),
                        mode='lines',
                        name='Prediction'))

fig_India.update_layout(title='Log scale of COVID-19 Cases Prediction in India')
fig_India.show()


# In[152]:


#save the graph

pio.write_html(fig_India, file=r'/home/songy4/Documents/fig_India.html', auto_open=True)


# In[ ]:


#UK


# In[153]:


#preprocess, timeseries generator

train_UK = df_UK

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_UK)
train_UK_scaled = scaler.transform(train_UK)


n_input = 90
n_features = 1
generator_UK = TimeseriesGenerator(train_UK_scaled,train_UK_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_UK, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

UK_list = list()
batch = train_UK_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    UK_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[UK_list[i]]],axis=1)


# In[154]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_UK.png');

#predict with lstm

UK_list = list()
batch = train_UK_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    UK_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[UK_list[i]]],axis=1)


# In[155]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_UK.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_UK_predict = pd.DataFrame(scaler.inverse_transform(UK_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_UK_predict= pd.concat([future_date, df_UK_predict], axis=1)
df_UK_predict= df_UK_predict.set_index('Date')
df_UK_proj = pd.concat([df_UK, df_UK_predict], axis=1)


df_UK_proj.index=pd.to_datetime(df_UK_proj.index)


# In[177]:


#plot numbers as log scale

fig_UK = go.Figure()
fig_UK.add_trace(go.Scatter(x=df_UK_proj.index, 
                        y=df_UK_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_UK.add_trace(go.Scatter(x=df_UK_proj.index, 
                        y=df_UK_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_UK.update_layout(title='COVID-19 Cases Prediction in UK')
fig_UK.show()


# In[178]:


#save the graph

pio.write_html(fig_UK, file=r'/home/songy4/Documents/fig_UK.html', auto_open=True)


# In[ ]:


#Peru


# In[158]:


#preprocess, timeseries generator

train_Peru = df_Peru

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_Peru)
train_Peru_scaled = scaler.transform(train_Peru)


n_input = 90
n_features = 1
generator_Peru = TimeseriesGenerator(train_Peru_scaled,train_Peru_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_Peru, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

Peru_list = list()
batch = train_Peru_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Peru_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Peru_list[i]]],axis=1)


# In[159]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_Peru.png');

#predict with lstm

Peru_list = list()
batch = train_Peru_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Peru_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Peru_list[i]]],axis=1)


# In[160]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_Peru.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_Peru_predict = pd.DataFrame(scaler.inverse_transform(Peru_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_Peru_predict= pd.concat([future_date, df_Peru_predict], axis=1)
df_Peru_predict= df_Peru_predict.set_index('Date')
df_Peru_proj = pd.concat([df_Peru, df_Peru_predict], axis=1)


df_Peru_proj.index=pd.to_datetime(df_Peru_proj.index)


# In[179]:


#plot numbers as log scale

fig_Peru = go.Figure()
fig_Peru.add_trace(go.Scatter(x=df_Peru_proj.index, 
                        y=df_Peru_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_Peru.add_trace(go.Scatter(x=df_Peru_proj.index, 
                        y=df_Peru_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_Peru.update_layout(title='COVID-19 Cases Prediction in Peru')
fig_Peru.show()


# In[180]:


#save the graph

pio.write_html(fig_Peru, file=r'/home/songy4/Documents/fig_Peru.html', auto_open=True)


# In[ ]:


#Chile


# In[181]:


#preprocess, timeseries generator

train_Chile = df_Chile

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_Chile)
train_Chile_scaled = scaler.transform(train_Chile)


n_input = 90
n_features = 1
generator_Chile = TimeseriesGenerator(train_Chile_scaled,train_Chile_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_Chile, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

Chile_list = list()
batch = train_Chile_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Chile_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Chile_list[i]]],axis=1)


# In[182]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_Chile.png');

#predict with lstm

Chile_list = list()
batch = train_Chile_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Chile_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Chile_list[i]]],axis=1)


# In[183]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_Chile.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_Chile_predict = pd.DataFrame(scaler.inverse_transform(Chile_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_Chile_predict= pd.concat([future_date, df_Chile_predict], axis=1)
df_Chile_predict= df_Chile_predict.set_index('Date')
df_Chile_proj = pd.concat([df_Chile, df_Chile_predict], axis=1)


df_Chile_proj.index=pd.to_datetime(df_Chile_proj.index)


# In[186]:


#plot numbers as log scale

fig_Chile = go.Figure()
fig_Chile.add_trace(go.Scatter(x=df_Chile_proj.index, 
                        y=df_Chile_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_Chile.add_trace(go.Scatter(x=df_Chile_proj.index, 
                        y=df_Chile_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_Chile.update_layout(title='COVID-19 Cases Prediction in Chile')
fig_Chile.show()


# In[187]:


#save the graph

pio.write_html(fig_Chile, file=r'/home/songy4/Documents/fig_Chile.html', auto_open=True)


# In[ ]:


#Spain


# In[294]:


#preprocess, timeseries generator

train_Spain = df_Spain

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_Spain)
train_Spain_scaled = scaler.transform(train_Spain)


n_input = 90
n_features = 1
generator_Spain = TimeseriesGenerator(train_Spain_scaled,train_Spain_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_Spain, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

Spain_list = list()
batch = train_Spain_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Spain_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Spain_list[i]]],axis=1)


# In[295]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_Spain.png');

#predict with lstm

Spain_list = list()
batch = train_Spain_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Spain_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Spain_list[i]]],axis=1)


# In[296]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_Spain.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_Spain_predict = pd.DataFrame(scaler.inverse_transform(Spain_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_Spain_predict= pd.concat([future_date, df_Spain_predict], axis=1)
df_Spain_predict= df_Spain_predict.set_index('Date')
df_Spain_proj = pd.concat([df_Spain, df_Spain_predict], axis=1)


df_Spain_proj.index=pd.to_datetime(df_Spain_proj.index)


# In[297]:


#plot numbers as log scale

fig_Spain = go.Figure()
fig_Spain.add_trace(go.Scatter(x=df_Spain_proj.index, 
                        y=df_Spain_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_Spain.add_trace(go.Scatter(x=df_Spain_proj.index, 
                        y=df_Spain_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_Spain.update_layout(title='COVID-19 Cases Prediction in Spain')
fig_Spain.show()


# In[298]:


#save the graph

pio.write_html(fig_Spain, file=r'/home/songy4/Documents/fig_Spain.html', auto_open=True)


# In[ ]:


#Italy


# In[194]:


#preprocess, timeseries generator

train_Italy = df_Italy

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_Italy)
train_Italy_scaled = scaler.transform(train_Italy)


n_input = 90
n_features = 1
generator_Italy = TimeseriesGenerator(train_Italy_scaled,train_Italy_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_Italy, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

Italy_list = list()
batch = train_Italy_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Italy_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Italy_list[i]]],axis=1)


# In[195]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_Italy.png');

#predict with lstm

Italy_list = list()
batch = train_Italy_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Italy_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Italy_list[i]]],axis=1)


# In[196]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_Italy.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_Italy_predict = pd.DataFrame(scaler.inverse_transform(Italy_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_Italy_predict= pd.concat([future_date, df_Italy_predict], axis=1)
df_Italy_predict= df_Italy_predict.set_index('Date')
df_Italy_proj = pd.concat([df_Italy, df_Italy_predict], axis=1)


df_Italy_proj.index=pd.to_datetime(df_Italy_proj.index)


# In[198]:


#plot numbers as log scale

fig_Italy = go.Figure()
fig_Italy.add_trace(go.Scatter(x=df_Italy_proj.index, 
                        y=df_Italy_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_Italy.add_trace(go.Scatter(x=df_Italy_proj.index, 
                        y=df_Italy_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_Italy.update_layout(title='COVID-19 Cases Prediction in Italy')
fig_Italy.show()


# In[199]:


#save the graph

pio.write_html(fig_Italy, file=r'/home/songy4/Documents/fig_Italy.html', auto_open=True)


# In[ ]:


#Iran


# In[200]:


#preprocess, timeseries generator

train_Iran = df_Iran

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_Iran)
train_Iran_scaled = scaler.transform(train_Iran)


n_input = 90
n_features = 1
generator_Iran = TimeseriesGenerator(train_Iran_scaled,train_Iran_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_Iran, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

Iran_list = list()
batch = train_Iran_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Iran_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Iran_list[i]]],axis=1)


# In[201]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_Iran.png');

#predict with lstm

Iran_list = list()
batch = train_Iran_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Iran_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Iran_list[i]]],axis=1)


# In[202]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_Iran.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_Iran_predict = pd.DataFrame(scaler.inverse_transform(Iran_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_Iran_predict= pd.concat([future_date, df_Iran_predict], axis=1)
df_Iran_predict= df_Iran_predict.set_index('Date')
df_Iran_proj = pd.concat([df_Iran, df_Iran_predict], axis=1)


df_Iran_proj.index=pd.to_datetime(df_Iran_proj.index)


# In[204]:


#plot numbers as log scale

fig_Iran = go.Figure()
fig_Iran.add_trace(go.Scatter(x=df_Iran_proj.index, 
                        y=df_Iran_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_Iran.add_trace(go.Scatter(x=df_Iran_proj.index, 
                        y=df_Iran_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_Iran.update_layout(title='COVID-19 Cases Prediction in Iran')
fig_Iran.show()


# In[205]:


#save the graph

pio.write_html(fig_Iran, file=r'/home/songy4/Documents/fig_Iran.html', auto_open=True)


# In[ ]:


#France


# In[299]:


#preprocess, timeseries generator

train_France = df_France

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_France)
train_France_scaled = scaler.transform(train_France)


n_input = 90
n_features = 1
generator_France = TimeseriesGenerator(train_France_scaled,train_France_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_France, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

France_list = list()
batch = train_France_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    France_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[France_list[i]]],axis=1)


# In[207]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_France.png');

#predict with lstm

France_list = list()
batch = train_France_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    France_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[France_list[i]]],axis=1)


# In[300]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_France.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_France_predict = pd.DataFrame(scaler.inverse_transform(France_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_France_predict= pd.concat([future_date, df_France_predict], axis=1)
df_France_predict= df_France_predict.set_index('Date')
df_France_proj = pd.concat([df_France, df_France_predict], axis=1)


df_France_proj.index=pd.to_datetime(df_France_proj.index)


# In[301]:


#plot numbers as log scale

fig_France = go.Figure()
fig_France.add_trace(go.Scatter(x=df_France_proj.index, 
                        y=df_France_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_France.add_trace(go.Scatter(x=df_France_proj.index, 
                        y=df_France_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_France.update_layout(title='COVID-19 Cases Prediction in France')
fig_France.show()


# In[302]:


#save the graph

pio.write_html(fig_France, file=r'/home/songy4/Documents/fig_France.html', auto_open=True)


# In[ ]:


#Germany


# In[212]:


#preprocess, timeseries generator

train_Germany = df_Germany

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_Germany)
train_Germany_scaled = scaler.transform(train_Germany)


n_input = 90
n_features = 1
generator_Germany = TimeseriesGenerator(train_Germany_scaled,train_Germany_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_Germany, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

Germany_list = list()
batch = train_Germany_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Germany_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Germany_list[i]]],axis=1)


# In[213]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_Germany.png');

#predict with lstm

Germany_list = list()
batch = train_Germany_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Germany_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Germany_list[i]]],axis=1)


# In[214]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_Germany.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_Germany_predict = pd.DataFrame(scaler.inverse_transform(Germany_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_Germany_predict= pd.concat([future_date, df_Germany_predict], axis=1)
df_Germany_predict= df_Germany_predict.set_index('Date')
df_Germany_proj = pd.concat([df_Germany, df_Germany_predict], axis=1)


df_Germany_proj.index=pd.to_datetime(df_Germany_proj.index)


# In[216]:


#plot numbers as log scale

fig_Germany = go.Figure()
fig_Germany.add_trace(go.Scatter(x=df_Germany_proj.index, 
                        y=df_Germany_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_Germany.add_trace(go.Scatter(x=df_Germany_proj.index, 
                        y=df_Germany_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_Germany.update_layout(title='COVID-19 Cases Prediction in Germany')
fig_Germany.show()


# In[217]:


#save the graph

pio.write_html(fig_Germany, file=r'/home/songy4/Documents/fig_Germany.html', auto_open=True)


# In[ ]:


#Turkey


# In[218]:


#preprocess, timeseries generator

train_Turkey = df_Turkey

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_Turkey)
train_Turkey_scaled = scaler.transform(train_Turkey)


n_input = 90
n_features = 1
generator_Turkey = TimeseriesGenerator(train_Turkey_scaled,train_Turkey_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_Turkey, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

Turkey_list = list()
batch = train_Turkey_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Turkey_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Turkey_list[i]]],axis=1)


# In[219]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_Turkey.png');

#predict with lstm

Turkey_list = list()
batch = train_Turkey_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Turkey_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Turkey_list[i]]],axis=1)


# In[220]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_Turkey.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_Turkey_predict = pd.DataFrame(scaler.inverse_transform(Turkey_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_Turkey_predict= pd.concat([future_date, df_Turkey_predict], axis=1)
df_Turkey_predict= df_Turkey_predict.set_index('Date')
df_Turkey_proj = pd.concat([df_Turkey, df_Turkey_predict], axis=1)


df_Turkey_proj.index=pd.to_datetime(df_Turkey_proj.index)


# In[221]:


#plot numbers as log scale

fig_Turkey = go.Figure()
fig_Turkey.add_trace(go.Scatter(x=df_Turkey_proj.index, 
                        y=df_Turkey_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_Turkey.add_trace(go.Scatter(x=df_Turkey_proj.index, 
                        y=df_Turkey_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_Turkey.update_layout(title='COVID-19 Cases Prediction in Turkey')
fig_Turkey.show()


# In[222]:


#save the graph

pio.write_html(fig_Turkey, file=r'/home/songy4/Documents/fig_Turkey.html', auto_open=True)


# In[ ]:


#Mexico


# In[223]:


#preprocess, timeseries generator

train_Mexico = df_Mexico

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_Mexico)
train_Mexico_scaled = scaler.transform(train_Mexico)


n_input = 90
n_features = 1
generator_Mexico = TimeseriesGenerator(train_Mexico_scaled,train_Mexico_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_Mexico, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

Mexico_list = list()
batch = train_Mexico_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Mexico_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Mexico_list[i]]],axis=1)


# In[224]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_Mexico.png');

#predict with lstm

Mexico_list = list()
batch = train_Mexico_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Mexico_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Mexico_list[i]]],axis=1)


# In[225]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_Mexico.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_Mexico_predict = pd.DataFrame(scaler.inverse_transform(Mexico_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_Mexico_predict= pd.concat([future_date, df_Mexico_predict], axis=1)
df_Mexico_predict= df_Mexico_predict.set_index('Date')
df_Mexico_proj = pd.concat([df_Mexico, df_Mexico_predict], axis=1)


df_Mexico_proj.index=pd.to_datetime(df_Mexico_proj.index)


# In[227]:


#plot numbers as log scale

fig_Mexico = go.Figure()
fig_Mexico.add_trace(go.Scatter(x=df_Mexico_proj.index, 
                        y=df_Mexico_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_Mexico.add_trace(go.Scatter(x=df_Mexico_proj.index, 
                        y=df_Mexico_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_Mexico.update_layout(title='COVID-19 Cases Prediction in Mexico')
fig_Mexico.show()


# In[228]:


#save the graph

pio.write_html(fig_Mexico, file=r'/home/songy4/Documents/fig_Mexico.html', auto_open=True)


# In[ ]:


#South_Africa


# In[229]:


#preprocess, timeseries generator

train_South_Africa = df_South_Africa

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_South_Africa)
train_South_Africa_scaled = scaler.transform(train_South_Africa)


n_input = 90
n_features = 1
generator_South_Africa = TimeseriesGenerator(train_South_Africa_scaled,train_South_Africa_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_South_Africa, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

South_Africa_list = list()
batch = train_South_Africa_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    South_Africa_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[South_Africa_list[i]]],axis=1)


# In[230]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_South_Africa.png');

#predict with lstm

South_Africa_list = list()
batch = train_South_Africa_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    South_Africa_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[South_Africa_list[i]]],axis=1)


# In[234]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_South_Africa.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_South_Africa_predict = pd.DataFrame(scaler.inverse_transform(South_Africa_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_South_Africa_predict= pd.concat([future_date, df_South_Africa_predict], axis=1)
df_South_Africa_predict= df_South_Africa_predict.set_index('Date')
df_South_Africa_proj = pd.concat([df_South_Africa, df_South_Africa_predict], axis=1)


df_South_Africa_proj.index=pd.to_datetime(df_South_Africa_proj.index)


# In[235]:


#plot numbers as log scale

fig_South_Africa = go.Figure()
fig_South_Africa.add_trace(go.Scatter(x=df_South_Africa_proj.index, 
                        y=df_South_Africa_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_South_Africa.add_trace(go.Scatter(x=df_South_Africa_proj.index, 
                        y=df_South_Africa_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_South_Africa.update_layout(title='COVID-19 Cases Prediction in South Africa')
fig_South_Africa.show()


# In[236]:


#save the graph

pio.write_html(fig_South_Africa, file=r'/home/songy4/Documents/fig_South_Africa.html', auto_open=True)


# In[ ]:


#China


# In[303]:


#preprocess, timeseries generator

train_China = df_China

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_China)
train_China_scaled = scaler.transform(train_China)


n_input = 90
n_features = 1
generator_China = TimeseriesGenerator(train_China_scaled,train_China_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_China, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

China_list = list()
batch = train_China_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    China_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[China_list[i]]],axis=1)


# In[304]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_China.png');

#predict with lstm

China_list = list()
batch = train_China_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    China_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[China_list[i]]],axis=1)


# In[305]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_China.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_China_predict = pd.DataFrame(scaler.inverse_transform(China_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_China_predict= pd.concat([future_date, df_China_predict], axis=1)
df_China_predict= df_China_predict.set_index('Date')
df_China_proj = pd.concat([df_China, df_China_predict], axis=1)


df_China_proj.index=pd.to_datetime(df_China_proj.index)


# In[306]:


#plot numbers as log scale

fig_China = go.Figure()
fig_China.add_trace(go.Scatter(x=df_China_proj.index, 
                        y=df_China_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_China.add_trace(go.Scatter(x=df_China_proj.index, 
                        y=df_China_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_China.update_layout(title='COVID-19 Cases Prediction in China')
fig_China.show()


# In[307]:


#save the graph

pio.write_html(fig_China, file=r'/home/songy4/Documents/fig_China.html', auto_open=True)


# In[ ]:


#Sweden


# In[243]:


#preprocess, timeseries generator

train_Sweden = df_Sweden

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_Sweden)
train_Sweden_scaled = scaler.transform(train_Sweden)


n_input = 90
n_features = 1
generator_Sweden = TimeseriesGenerator(train_Sweden_scaled,train_Sweden_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_Sweden, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

Sweden_list = list()
batch = train_Sweden_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Sweden_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Sweden_list[i]]],axis=1)


# In[244]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_Sweden.png');

#predict with lstm

Sweden_list = list()
batch = train_Sweden_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Sweden_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Sweden_list[i]]],axis=1)


# In[245]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_Sweden.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_Sweden_predict = pd.DataFrame(scaler.inverse_transform(Sweden_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_Sweden_predict= pd.concat([future_date, df_Sweden_predict], axis=1)
df_Sweden_predict= df_Sweden_predict.set_index('Date')
df_Sweden_proj = pd.concat([df_Sweden, df_Sweden_predict], axis=1)


df_Sweden_proj.index=pd.to_datetime(df_Sweden_proj.index)


# In[246]:


#plot numbers as log scale

fig_Sweden = go.Figure()
fig_Sweden.add_trace(go.Scatter(x=df_Sweden_proj.index, 
                        y=df_Sweden_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_Sweden.add_trace(go.Scatter(x=df_Sweden_proj.index, 
                        y=df_Sweden_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_Sweden.update_layout(title='COVID-19 Cases Prediction in Sweden')
fig_Sweden.show()


# In[247]:


#save the graph

pio.write_html(fig_Sweden, file=r'/home/songy4/Documents/fig_Sweden.html', auto_open=True)


# In[ ]:


#Singapore


# In[308]:


#preprocess, timeseries generator

train_Singapore = df_Singapore

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_Singapore)
train_Singapore_scaled = scaler.transform(train_Singapore)


n_input = 90
n_features = 1
generator_Singapore = TimeseriesGenerator(train_Singapore_scaled,train_Singapore_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_Singapore, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

Singapore_list = list()
batch = train_Singapore_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Singapore_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Singapore_list[i]]],axis=1)


# In[309]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_Singapore.png');

#predict with lstm

Singapore_list = list()
batch = train_Singapore_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Singapore_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Singapore_list[i]]],axis=1)


# In[310]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_Singapore.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_Singapore_predict = pd.DataFrame(scaler.inverse_transform(Singapore_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_Singapore_predict= pd.concat([future_date, df_Singapore_predict], axis=1)
df_Singapore_predict= df_Singapore_predict.set_index('Date')
df_Singapore_proj = pd.concat([df_Singapore, df_Singapore_predict], axis=1)


df_Singapore_proj.index=pd.to_datetime(df_Singapore_proj.index)


# In[311]:


#plot numbers as log scale

fig_Singapore = go.Figure()
fig_Singapore.add_trace(go.Scatter(x=df_Singapore_proj.index, 
                        y=df_Singapore_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_Singapore.add_trace(go.Scatter(x=df_Singapore_proj.index, 
                        y=df_Singapore_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_Singapore.update_layout(title='COVID-19 Cases Prediction in Singapore')
fig_Singapore.show()


# In[312]:


#save the graph

pio.write_html(fig_Singapore, file=r'/home/songy4/Documents/fig_Singapore.html', auto_open=True)


# In[ ]:


#South Korea


# In[253]:


#preprocess, timeseries generator

train_South_Korea = df_South_Korea

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_South_Korea)
train_South_Korea_scaled = scaler.transform(train_South_Korea)


n_input = 90
n_features = 1
generator_South_Korea = TimeseriesGenerator(train_South_Korea_scaled,train_South_Korea_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_South_Korea, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

South_Korea_list = list()
batch = train_South_Korea_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    South_Korea_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[South_Korea_list[i]]],axis=1)


# In[254]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_South_Korea.png');

#predict with lstm

South_Korea_list = list()
batch = train_South_Korea_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    South_Korea_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[South_Korea_list[i]]],axis=1)


# In[255]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_South_Korea.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_South_Korea_predict = pd.DataFrame(scaler.inverse_transform(South_Korea_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_South_Korea_predict= pd.concat([future_date, df_South_Korea_predict], axis=1)
df_South_Korea_predict= df_South_Korea_predict.set_index('Date')
df_South_Korea_proj = pd.concat([df_South_Korea, df_South_Korea_predict], axis=1)


df_South_Korea_proj.index=pd.to_datetime(df_South_Korea_proj.index)


# In[256]:


#plot numbers as log scale

fig_South_Korea = go.Figure()
fig_South_Korea.add_trace(go.Scatter(x=df_South_Korea_proj.index, 
                        y=df_South_Korea_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_South_Korea.add_trace(go.Scatter(x=df_South_Korea_proj.index, 
                        y=df_South_Korea_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_South_Korea.update_layout(title='COVID-19 Cases Prediction in South Korea')
fig_South_Korea.show()


# In[257]:


#save the graph

pio.write_html(fig_South_Korea, file=r'/home/songy4/Documents/fig_South_Korea.html', auto_open=True)


# In[ ]:


#Slovenia


# In[258]:


#preprocess, timeseries generator

train_Slovenia = df_Slovenia

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_Slovenia)
train_Slovenia_scaled = scaler.transform(train_Slovenia)


n_input = 90
n_features = 1
generator_Slovenia = TimeseriesGenerator(train_Slovenia_scaled,train_Slovenia_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_Slovenia, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

Slovenia_list = list()
batch = train_Slovenia_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Slovenia_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Slovenia_list[i]]],axis=1)


# In[259]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_Slovenia.png');

#predict with lstm

Slovenia_list = list()
batch = train_Slovenia_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Slovenia_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Slovenia_list[i]]],axis=1)


# In[260]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_Slovenia.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_Slovenia_predict = pd.DataFrame(scaler.inverse_transform(Slovenia_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_Slovenia_predict= pd.concat([future_date, df_Slovenia_predict], axis=1)
df_Slovenia_predict= df_Slovenia_predict.set_index('Date')
df_Slovenia_proj = pd.concat([df_Slovenia, df_Slovenia_predict], axis=1)


df_Slovenia_proj.index=pd.to_datetime(df_Slovenia_proj.index)


# In[261]:


#plot numbers as log scale

fig_Slovenia = go.Figure()
fig_Slovenia.add_trace(go.Scatter(x=df_Slovenia_proj.index, 
                        y=df_Slovenia_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_Slovenia.add_trace(go.Scatter(x=df_Slovenia_proj.index, 
                        y=df_Slovenia_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_Slovenia.update_layout(title='COVID-19 Cases Prediction in Slovenia')
fig_Slovenia.show()


# In[262]:


#save the graph

pio.write_html(fig_Slovenia, file=r'/home/songy4/Documents/fig_Slovenia.html', auto_open=True)


# In[ ]:


#New Zealand


# In[263]:


#preprocess, timeseries generator

train_New_Zealand = df_New_Zealand

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_New_Zealand)
train_New_Zealand_scaled = scaler.transform(train_New_Zealand)


n_input = 90
n_features = 1
generator_New_Zealand = TimeseriesGenerator(train_New_Zealand_scaled,train_New_Zealand_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_New_Zealand, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

New_Zealand_list = list()
batch = train_New_Zealand_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    New_Zealand_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[New_Zealand_list[i]]],axis=1)


# In[264]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_New_Zealand.png');

#predict with lstm

New_Zealand_list = list()
batch = train_New_Zealand_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    New_Zealand_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[New_Zealand_list[i]]],axis=1)


# In[266]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_New_Zealand.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_New_Zealand_predict = pd.DataFrame(scaler.inverse_transform(New_Zealand_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_New_Zealand_predict= pd.concat([future_date, df_New_Zealand_predict], axis=1)
df_New_Zealand_predict= df_New_Zealand_predict.set_index('Date')
df_New_Zealand_proj = pd.concat([df_New_Zealand, df_New_Zealand_predict], axis=1)


df_New_Zealand_proj.index=pd.to_datetime(df_New_Zealand_proj.index)


# In[267]:


#plot numbers as log scale

fig_New_Zealand = go.Figure()
fig_New_Zealand.add_trace(go.Scatter(x=df_New_Zealand_proj.index, 
                        y=df_New_Zealand_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_New_Zealand.add_trace(go.Scatter(x=df_New_Zealand_proj.index, 
                        y=df_New_Zealand_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_New_Zealand.update_layout(title='COVID-19 Cases Prediction in New Zealand')
fig_New_Zealand.show()


# In[268]:


#save the graph

pio.write_html(fig_New_Zealand, file=r'/home/songy4/Documents/fig_New_Zealand.html', auto_open=True)


# In[ ]:


#Vietnam


# In[269]:


#preprocess, timeseries generator

train_Vietnam = df_Vietnam

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_Vietnam)
train_Vietnam_scaled = scaler.transform(train_Vietnam)


n_input = 90
n_features = 1
generator_Vietnam = TimeseriesGenerator(train_Vietnam_scaled,train_Vietnam_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_Vietnam, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

Vietnam_list = list()
batch = train_Vietnam_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Vietnam_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Vietnam_list[i]]],axis=1)


# In[270]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_Vietnam.png');

#predict with lstm

Vietnam_list = list()
batch = train_Vietnam_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Vietnam_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Vietnam_list[i]]],axis=1)


# In[271]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_Vietnam.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_Vietnam_predict = pd.DataFrame(scaler.inverse_transform(Vietnam_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_Vietnam_predict= pd.concat([future_date, df_Vietnam_predict], axis=1)
df_Vietnam_predict= df_Vietnam_predict.set_index('Date')
df_Vietnam_proj = pd.concat([df_Vietnam, df_Vietnam_predict], axis=1)


df_Vietnam_proj.index=pd.to_datetime(df_Vietnam_proj.index)


# In[273]:


#plot numbers as log scale

fig_Vietnam = go.Figure()
fig_Vietnam.add_trace(go.Scatter(x=df_Vietnam_proj.index, 
                        y=np.log10(df_Vietnam_proj.Total_case+1),
                        mode='lines',
                        name='Actual'))
fig_Vietnam.add_trace(go.Scatter(x=df_Vietnam_proj.index, 
                        y=np.log10(df_Vietnam_proj.Prediction),
                        mode='lines',
                        name='Prediction'))

fig_Vietnam.update_layout(title='Log scale of COVID-19 Cases Prediction in Vietnam')
fig_Vietnam.show()


# In[274]:


#save the graph

pio.write_html(fig_Vietnam, file=r'/home/songy4/Documents/fig_Vietnam.html', auto_open=True)


# In[ ]:


#Taiwan


# In[313]:


#preprocess, timeseries generator

train_Taiwan = df_Taiwan

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_Taiwan)
train_Taiwan_scaled = scaler.transform(train_Taiwan)


n_input = 90
n_features = 1
generator_Taiwan = TimeseriesGenerator(train_Taiwan_scaled,train_Taiwan_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_Taiwan, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

Taiwan_list = list()
batch = train_Taiwan_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Taiwan_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Taiwan_list[i]]],axis=1)


# In[314]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_Taiwan.png');

#predict with lstm

Taiwan_list = list()
batch = train_Taiwan_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Taiwan_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Taiwan_list[i]]],axis=1)


# In[315]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_Taiwan.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_Taiwan_predict = pd.DataFrame(scaler.inverse_transform(Taiwan_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_Taiwan_predict= pd.concat([future_date, df_Taiwan_predict], axis=1)
df_Taiwan_predict= df_Taiwan_predict.set_index('Date')
df_Taiwan_proj = pd.concat([df_Taiwan, df_Taiwan_predict], axis=1)


df_Taiwan_proj.index=pd.to_datetime(df_Taiwan_proj.index)


# In[316]:


#plot numbers as log scale

fig_Taiwan = go.Figure()
fig_Taiwan.add_trace(go.Scatter(x=df_Taiwan_proj.index, 
                        y=df_Taiwan_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_Taiwan.add_trace(go.Scatter(x=df_Taiwan_proj.index, 
                        y=df_Taiwan_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_Taiwan.update_layout(title='COVID-19 Cases Prediction in Taiwan')
fig_Taiwan.show()


# In[317]:


#save the graph

pio.write_html(fig_Taiwan, file=r'/home/songy4/Documents/fig_Taiwan.html', auto_open=True)


# In[ ]:


#Cambodia


# In[280]:


#preprocess, timeseries generator

train_Cambodia = df_Cambodia

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_Cambodia)
train_Cambodia_scaled = scaler.transform(train_Cambodia)


n_input = 90
n_features = 1
generator_Cambodia = TimeseriesGenerator(train_Cambodia_scaled,train_Cambodia_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_Cambodia, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

Cambodia_list = list()
batch = train_Cambodia_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Cambodia_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Cambodia_list[i]]],axis=1)


# In[281]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_Cambodia.png');

#predict with lstm

Cambodia_list = list()
batch = train_Cambodia_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Cambodia_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Cambodia_list[i]]],axis=1)


# In[282]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_Cambodia.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_Cambodia_predict = pd.DataFrame(scaler.inverse_transform(Cambodia_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_Cambodia_predict= pd.concat([future_date, df_Cambodia_predict], axis=1)
df_Cambodia_predict= df_Cambodia_predict.set_index('Date')
df_Cambodia_proj = pd.concat([df_Cambodia, df_Cambodia_predict], axis=1)


df_Cambodia_proj.index=pd.to_datetime(df_Cambodia_proj.index)


# In[284]:


#plot numbers as log scale

fig_Cambodia = go.Figure()
fig_Cambodia.add_trace(go.Scatter(x=df_Cambodia_proj.index, 
                        y=df_Cambodia_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_Cambodia.add_trace(go.Scatter(x=df_Cambodia_proj.index, 
                        y=df_Cambodia_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_Cambodia.update_layout(title='COVID-19 Cases Prediction in Cambodia')
fig_Cambodia.show()


# In[285]:


#save the graph

pio.write_html(fig_Cambodia, file=r'/home/songy4/Documents/fig_Cambodia.html', auto_open=True)


# In[ ]:


#Laos


# In[286]:


#preprocess, timeseries generator

train_Laos = df_Laos

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(train_Laos)
train_Laos_scaled = scaler.transform(train_Laos)


n_input = 90
n_features = 1
generator_Laos = TimeseriesGenerator(train_Laos_scaled,train_Laos_scaled, 
                                n_input, batch_size=30)
        

lstm_model.fit_generator(generator_Laos, steps_per_epoch=1, epochs=100,
                        verbose=1)
#predict with lstm

Laos_list = list()
batch = train_Laos_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Laos_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Laos_list[i]]],axis=1)


# In[287]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss_Laos.png');

#predict with lstm

Laos_list = list()
batch = train_Laos_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    Laos_list.append(lstm_model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[Laos_list[i]]],axis=1)


# In[288]:


#concat future predict to new dataframe

future_date = pd.DataFrame({'Date':pd.date_range(start=df_Laos.index[-1], 
                                                 periods=91, freq='D', closed='right')})
df_Laos_predict = pd.DataFrame(scaler.inverse_transform(Laos_list), 
                             index=future_date[-n_input:].index, 
                             columns = ['Prediction'])
df_Laos_predict= pd.concat([future_date, df_Laos_predict], axis=1)
df_Laos_predict= df_Laos_predict.set_index('Date')
df_Laos_proj = pd.concat([df_Laos, df_Laos_predict], axis=1)


df_Laos_proj.index=pd.to_datetime(df_Laos_proj.index)


# In[290]:


#plot numbers as log scale

fig_Laos = go.Figure()
fig_Laos.add_trace(go.Scatter(x=df_Laos_proj.index, 
                        y=df_Laos_proj.Total_case,
                        mode='lines',
                        name='Actual'))
fig_Laos.add_trace(go.Scatter(x=df_Laos_proj.index, 
                        y=df_Laos_proj.Prediction,
                        mode='lines',
                        name='Prediction'))

fig_Laos.update_layout(title='COVID-19 Cases Prediction in Laos')
fig_Laos.show()


# In[291]:


#save the graph

pio.write_html(fig_Laos, file=r'/home/songy4/Documents/fig_Laos.html', auto_open=True)


# In[331]:


#all figures to one html

def figures_to_html(figs, filename=f"{os.path.join(os.path.sep,'home', 'songy4', 'Documents','dashboard.html')}"):
    dashboard=open(filename, 'w')
    dashboard.write("<html><head></head><body>" + "\n")
    for fig in figs:
        inner_html = fig.to_html().split('<body')[1].split('</body>')[0]
        dashboard.write(inner_html)
    dashboard.write("</body></html>" + "\n")
    
figures_to_html([fig_US, fig_Brazil, fig_Russia, fig_India, fig_UK, 
                 fig_Peru, fig_Chile, fig_Spain, fig_Italy, fig_Iran, 
                 fig_France, fig_Germany, fig_Turkey, fig_Mexico, 
                 fig_South_Africa, fig_China, fig_Sweden, fig_Singapore, 
                 fig_South_Korea, fig_Slovenia, fig_New_Zealand, fig_Vietnam, 
                 fig_Taiwan, fig_Cambodia, fig_Laos])
    

