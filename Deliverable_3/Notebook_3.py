
# coding: utf-8

# In[1]:


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
from arch.unitroot import ADF


# In[2]:


#put random seed
sd=1
np.random.seed(sd)
rn.seed(sd)
os.environ['PYTHONHASHSEED'] = str(sd)


# In[3]:


#set environment for consistency

from keras import backend as K

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.random.set_seed(sd)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config = config)
tf.compat.v1.keras.backend.set_session(sess)


# In[4]:


#Load the global confirmed cases by date datasets
case = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df_case = pd.read_csv(case)
df_case.head()


# In[5]:


#Drop a Province/State, Lat, Long columns, combine rows with same country name, change Country/Region to Country, and reset index

df_case = df_case.drop(df_case.columns[[0, 2, 3]], axis=1)
df_case = df_case.groupby(df_case['Country/Region']).aggregate('sum')
df_case.index.names = ['Country']
df_case.reset_index(level=0, inplace=True)
df_case.head()


# In[6]:


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


# In[7]:


#chech df_melt shape

df_melt.shape


# In[8]:


#Change few country names in the dataframe

df_melt = df_melt.replace({'Country':{'United Kingdom':'UK',
                                      'South Africa':'South_Africa', 
                                      'Korea, South':'South_Korea',
                                      'New Zealand':'New_Zealand', 
                                      'Taiwan*':'Taiwan'}})


# In[9]:


#Group by country among 25 countries

country_25 = ['US','Brazil','Russia','India','UK','Peru','Chile',
              'Spain','Italy','Iran','France','Germany', 'Turkey', 
              'Mexico','South_Africa','China','Sweden','Singapore', 
              'South_Korea', 'Slovenia','New_Zealand','Vietnam',
              'Taiwan','Cambodia','Laos']


# In[10]:


#loop to create dataframe for each country in country_25

gbl = globals()

for i in df_melt.Country:
    if i in country_25:
        gbl['df_'+i] = df_melt.loc[df_melt.Country == i]
        gbl['df_'+i] = gbl['df_'+i].drop('Country', axis=1)
        gbl['df_'+i] = gbl['df_'+i].set_index('Date')
        gbl['df_'+i].index = pd.to_datetime(gbl['df_'+i].index, format='%m/%d/%Y'). strftime('%Y-%m-%d')


# In[11]:


#Split 25 countries dataframes into train and test 80 and 20 ratio

gbl = globals()

for i in df_melt.Country:
    if i in country_25:
        size_i = int(len(gbl['df_'+i])*0.80)
        gbl['train_'+i] = gbl['df_'+i][:size_i]
        gbl['test_'+i] = gbl['df_'+i][size_i:]


# In[12]:


#visualize data split for US

x_US = train_US.values
y_US = test_US.values
pyplot.plot(x_US)
pyplot.plot([None for i in x_US] + [x_US for x_US in y_US])
pyplot.show()


# In[13]:


#Stack subplot the total case of 25 countries

fig_1, axs = plt.subplots(5, 5, figsize = (25, 25), sharex=True)
axs[0, 0].plot(df_US, 'tab:orange')
axs[0, 0].set_title('US')
axs[0, 1].plot(df_Brazil)
axs[0, 1].set_title('Brazil')
axs[0, 2].plot(df_Russia, 'tab:red')
axs[0, 2].set_title('Russia')
axs[0, 3].plot(df_India, 'tab:green')
axs[0, 3].set_title('India')
axs[0, 4].plot(df_UK, 'tab:red')
axs[0, 4].set_title('UK')
axs[1, 0].plot(df_Peru, 'tab:brown')
axs[1, 0].set_title('Peru')
axs[1, 1].plot(df_Chile, 'tab:brown')
axs[1, 1].set_title('Chile')
axs[1, 2].plot(df_Spain, 'tab:red')
axs[1, 2].set_title('Spain')
axs[1, 3].plot(df_Italy, 'tab:red')
axs[1, 3].set_title('Italy')
axs[1, 4].plot(df_Iran)
axs[1, 4].set_title('Iran')
axs[2, 0].plot(df_France, 'tab:red')
axs[2, 0].set_title('France')
axs[2, 1].plot(df_Germany, 'tab:red')
axs[2, 1].set_title('Germany')
axs[2, 2].plot(df_Turkey)
axs[2, 2].set_title('Turkey')
axs[2, 3].plot(df_Mexico, 'tab:brown')
axs[2, 3].set_title('Mexico')
axs[2, 4].plot(df_South_Africa)
axs[2, 4].set_title('South Africa')
axs[3, 0].plot(df_China, 'tab:green')
axs[3, 0].set_title('China')
axs[3, 1].plot(df_Sweden, 'tab:red')
axs[3, 1].set_title('Sweden')
axs[3, 2].plot(df_Singapore, 'tab:green')
axs[3, 2].set_title('Singapore')
axs[3, 3].plot(df_South_Korea, 'tab:green')
axs[3, 3].set_title('South Korea')
axs[3, 4].plot(df_Slovenia, 'tab:red')
axs[3, 4].set_title('Slovenia')
axs[4, 0].plot(df_New_Zealand)
axs[4, 0].set_title('New Zealand')
axs[4, 1].plot(df_Vietnam, 'tab:green')
axs[4, 1].set_title('Vietnam')
axs[4, 2].plot(df_Taiwan, 'tab:green')
axs[4, 2].set_title('Taiwan')
axs[4, 3].plot(df_Cambodia, 'tab:green')
axs[4, 3].set_title('Cambodia')
axs[4, 4].plot(df_Laos, 'tab:green')
axs[4, 4].set_title('Laos')

plt.setp(axs, xticks=[])


# In[14]:


#save the fig_1
fig_1.savefig(r'/home/songy4/Documents/subplot.png')


# Forecast by ARIMA 

# In[15]:


#Set the frequency of index datetime
df_US.index.freq='DS'


# In[16]:


#decompose data
fig_2= sm.tsa.seasonal_decompose(df_US, period=7, model='additive').plot()
fig_2.autofmt_xdate()
plt.show()


# In[17]:


fig_2.savefig(r'/home/songy4/Documents/decompose.png')


# In[18]:


#calculation for rolling statistics
df_1 = pd.DataFrame()
df_2 = pd.DataFrame()
df_1['z_data'] = (df_US['Total_case'] - df_US.Total_case.rolling(window=7).mean())/df_US.Total_case.rolling(window=7).std()
df_2['zp_data'] = df_1['z_data'] - df_1['z_data'].shift(10)


# In[19]:


#check stationarity by plotting the data above

fig_3, ax = plt.subplots(3, figsize = (25, 20))
ax[0].plot(df_US.index, df_US.Total_case, label='US Total Case')
ax[0].plot(df_US.Total_case.rolling(window=7).mean(), label='Rolling Mean')
ax[0].plot(df_US.Total_case.rolling(window=7).std(), label='Rolling Std (x10)')
ax[0].legend()

ax[1].plot(df_US.index, df_1.z_data, label='De-Trended Data')
ax[1].plot(df_1.z_data.rolling(window=7).mean(), label='Rolling Mean')
ax[1].plot(df_1.z_data.rolling(window=7).std(), label='Rolling Std (x10)')
ax[1].legend()

ax[2].plot(df_US.index, df_2.zp_data, label='10 Lag Differenced De-Trended Data')
ax[2].plot(df_2.zp_data.rolling(window=7).mean(), label='Rolling Mean')
ax[2].plot(df_2.zp_data.rolling(window=7).std(), label='Rolling Std (x10)')
ax[2].legend()

plt.tight_layout()
fig_3.autofmt_xdate()


# In[20]:


fig_3.savefig(r'/home/songy4/Documents/rolling.png')


# In[21]:


#adf test for raw data

adf = ADF(df_US)
print(adf.summary().as_text())


# In[22]:


#adf test for de-trended and 7-lag de-trended data to check stationary 
   
print('Check the stationary of the de-trneded data:')
dftest = adfuller(df_1.z_data.dropna(), autolag='AIC')
print('Test statistic = {:.3f}'.format(dftest[0]))
print('P-value = {:.3f}'.format(dftest[1]))
print('Critical values:')
for k, v in dftest[4].items():
    print('\t{}: {} - The data is {} stationary with {}% confidence'.format(k, v, 'not' 
                                                                            if v<dftest[0]                                                                          
                                                                            else '', 100-int(k[:-1])))

print('Check the stationary of the 10 lag differenced de-trended data:')
dftest = adfuller(df_2.zp_data.dropna(), autolag='AIC')
print('Test statistic = {:.3f}'.format(dftest[0]))
print('P-value = {:.3f}'.format(dftest[1]))
print('Critical values:')
for k, v in dftest[4].items():
    print('\t{}: {} - The data is {} stationary with {}% confidence'.format(k, v, 'not' 
                                                                            if v<dftest[0]
                                                                            else '', 100-int(k[:-1])))


# In[23]:


#differencing and autocorrelation: US

fig_4, axes = plt.subplots(3, 3, figsize = (25, 25))
axes[0, 0].plot(df_US); axes[0, 0].set_title('US Total Case')
plot_acf(df_US, ax=axes[0, 1])
autocorrelation_plot(df_US, ax=axes[0, 2])

#1st differencing
axes[1, 0].plot(df_US.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df_US.diff().dropna(), ax=axes[1, 1])
autocorrelation_plot(df_US.diff().dropna(), ax=axes[1, 2])

#2nd differencing
axes[2, 0].plot(df_US.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df_US.diff().diff().dropna(), ax=axes[2, 1])
autocorrelation_plot(df_US.diff().diff().dropna(), ax=axes[2, 2])

plt.setp(axes, xticks=[])
plt.show()


# In[24]:


fig_4.savefig(r'/home/songy4/Documents/differecing.png')


# In[25]:


#Test to # of differencing

#augmented Dickey-Fuller test
adf = ndiffs(df_US, test='adf')

#Kwiatowski-Phillips-Schmidt-Shin (KPSS) test
kpss = ndiffs(df_US, test='kpss')

#Phillips-Perron test
pp = ndiffs(df_US, test='pp')

print(f'ADF test: {adf} \n KPSS test: {kpss} \n PP test: {pp}')


# In[26]:


#partial autocorrelation function plot for 2nd differenced series

plt.rcParams.update({'figure.figsize':(9, 3), 'figure.dpi':120})

fig_5, axes =plt.subplots(1,2)
axes[0].plot(df_US.diff().diff()); axes[0].set_title('2nd Differencing')
axes[1].set(ylim=(0,1.2))
plot_pacf(df_US.diff().diff().dropna(), ax=axes[1])

plt.setp(axes[0], xticks=[])
plt.show()


# In[27]:


fig_5.savefig(r'/home/songy4/Documents/PACF.png')


# In[28]:


#autocorrelation function plot for 2nd differenced series

plt.rcParams.update({'figure.figsize':(9, 3), 'figure.dpi':120})

fig_6, axes =plt.subplots(1,2)
axes[0].plot(df_US.diff().diff()); axes[0].set_title('2nd Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(df_US.diff().diff().dropna(), ax=axes[1])

plt.setp(axes[0], xticks=[])
plt.show()


# In[29]:


fig_6.savefig(r'/home/songy4/Documents/ACF.png')


# In[30]:


#fit the model

arima = sm.tsa.ARIMA(train_US, order=(1, 2, 1)).fit(disp=False)
print(arima.summary())


# In[31]:


#predict with arima

arima_pred = arima.predict(start = len(train_US), 
                               end =len(df_US)-1, typ='levels').rename('ARIMA Predictions')
arima_pred


# In[32]:


#Add ARIMA Predictions to test_US

test_US.loc[:,('ARIMA_Predictions')] = arima_pred


# In[33]:


#plot the prediction vs actual

fig_7 = plt.figure(figsize = (25, 12))
ax = sns.lineplot(x=test_US.index, y=test_US['Total_case'], label='US Total Case')
sns.lineplot(x=test_US.index, y=test_US['ARIMA_Predictions'], label='ARIMA Predictions')
ax.legend()


# In[34]:


fig_7.savefig(r'/home/songy4/Documents/arima.png')


# In[35]:


#plot residual errors
residuals = pd.DataFrame(arima.resid)
residuals.plot()
plt.savefig(r'/home/songy4/Documents/residual_arima.png')
pyplot.show()
residuals.plot(kind='kde')
plt.savefig(r'/home/songy4/Documents/density_arima.png')
pyplot.show()
print(residuals.describe())


# In[36]:


#rmse error

arima_rmse_error = rmse(test_US['Total_case'], arima_pred)
arima_mse_error = arima_rmse_error**2
mean_value = df_US['Total_case'].mean()

print(f'RMSE Error: {arima_rmse_error} \n  MSE Error: {arima_mse_error} \n Mean: {mean_value}')


# In[37]:


#grab 95% confidence band with forecast

fc, se, conf = arima.forecast(len(test_US), 
                              alpha=0.05)

fc_series = pd.Series(fc, index=test_US.index)
lower_series = pd.Series(conf[:,0], index=test_US.index)
upper_series = pd.Series(conf[:,1], index=test_US.index)


# In[38]:


#plot overall data with the prediction with 95% band vs actual 

fig_8 = plt.figure(figsize = (25, 12), dpi=100)
plt.plot(train_US, label='Training')
plt.plot(test_US, label='Actual')
plt.plot(fc_series, label='Forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('COVID-19 Cases in US Actual vs ARIMA Forecast')
plt.legend(loc='upper left', fontsize=12)
fig_8.autofmt_xdate()
plt.show()


# In[39]:


fig_8.savefig(r'/home/songy4/Documents/arima_total.png')


# In[40]:


#accuracy metrics: MAPE, correlation, min-max error

arima_Mean_Forecast_Errors = np.mean(test_US.Total_case-fc_series)
arima_MAPE = np.mean(np.abs((arima_Mean_Forecast_Errors)/test_US.Total_case))*100
arima_Corr = np.corrcoef(fc_series, test_US.Total_case)[0,1]
arima_Min_Max_Error = 1 - np.mean(lower_series/upper_series)

print(f'Mean Forecast_error: {arima_Mean_Forecast_Errors} \n MAPE: {arima_MAPE} \n Correlation: {arima_Corr} \n Min Max Error: {arima_Min_Max_Error}')


# In[ ]:


Forecast by LSTM


# In[41]:


#preprocess using minmaxscaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_US)
train_US_scaled = scaler.transform(train_US)
test_US_scaled = scaler.transform(test_US)


# In[42]:


#create a time series generator object

from keras.preprocessing.sequence import TimeseriesGenerator

n_input = 90
n_features = 1
generator = TimeseriesGenerator(train_US_scaled, train_US_scaled, 
                                length=n_input, batch_size=20)


# In[56]:


#build keras lstm model

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

lstm_model = Sequential()
lstm_model.add(LSTM(125, activation='relu', return_sequences=True, input_shape=(n_input, n_features)))
lstm_model.add(LSTM(80, activation='relu', return_sequences=True))
lstm_model.add(LSTM(50, activation='relu'))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.summary()


# In[57]:


#fit the lstm model

lstm_model.fit_generator(generator, steps_per_epoch=1, epochs=100,
                        verbose=1)


# In[58]:


#plot the loss

losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(12,4))
plt.xticks(np.arange(0, 101, 25))
plt.plot(range(len(losses_lstm)), losses_lstm)

plt.savefig(r'/home/songy4/Documents/lstm_loss.png');


# In[59]:


#predict with lstm

lstm_predictions_US_scaled = list()
batch = train_US_scaled[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_US_scaled)):
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_US_scaled.append(lstm_pred)
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)


# In[60]:


#print scaled prediction

lstm_predictions_US_scaled


# In[61]:


#scale back the prediction

lstm_predictions = scaler.inverse_transform(lstm_predictions_US_scaled)


# In[62]:


#print prediction numbers
lstm_predictions


# In[63]:


#append the lstm_predictions on test_US

test_US.loc[:,('LSTM_Predictions')] = lstm_predictions
test_US


# In[64]:


#plot lstm

fig_9 = plt.figure(figsize = (25, 12))
ax = sns.lineplot(x=test_US.index, y=test_US['Total_case'], label='US Total Case')
sns.lineplot(x=test_US.index, y=test_US['LSTM_Predictions'], label='LSTM Predictions')
fig_9.autofmt_xdate()
ax.legend()


# In[65]:


#Save a plot
fig_9.savefig(r'/home/songy4/Documents/lstm.png')


# In[66]:


#Check rmse error

lstm_rmse_error = rmse(test_US['Total_case'], test_US['LSTM_Predictions'])
lstm_mse_error = lstm_rmse_error **2
mean_value = df_US['Total_case'].mean()

print(f'RMSE Error: {lstm_rmse_error} \n MSE Error: {lstm_mse_error} \n Mean: {mean_value}')


# In[67]:


#accuracy metrics: MAPE, correlation, min-max error

lstm_Mean_Forecast_Errors = np.mean(test_US.Total_case-test_US['LSTM_Predictions'])
lstm_MAPE = np.mean(np.abs((lstm_Mean_Forecast_Errors)/test_US.Total_case))*100
lstm_Corr = np.corrcoef(test_US.LSTM_Predictions, test_US.Total_case)[0,1]


print(f'Mean Forecast_error: {lstm_Mean_Forecast_Errors} \n MAPE: {lstm_MAPE} \n Correlation: {lstm_Corr}')


# Forecast by Prophet

# In[68]:


#Prepare dataset for prophet forecast

gbl = globals()

for i in df_melt.Country:
    if i in country_25:
        gbl['df_'+i+'_pr'] = gbl['df_'+i].copy()
        gbl['df_'+i+'_pr'] = gbl['df_'+i].reset_index()
        gbl['df_'+i+'_pr'].columns = ['ds', 'y']


# In[69]:


#make y as log scale and keep original y value as y_orig

gbl = globals()

for i in country_25:  
    gbl['df_'+i+'_pr'].loc[:, 'y_orig'] = gbl['df_'+i+'_pr']['y']
    gbl['df_'+i+'_pr']['y']=gbl['df_'+i+'_pr']['y'].apply(lambda x: 0 if x==0 else np.log(x))


# In[70]:


#change column name and split train and test

gbl = globals()

for i in df_melt.Country:
    if i in country_25:
        size_i = int(len(gbl['df_'+i+'_pr'])*0.80)
        gbl['train_'+i+'_pr'] = gbl['df_'+i+'_pr'][:size_i]
        gbl['test_'+i+'_pr'] = gbl['df_'+i+'_pr'][size_i:]


# In[71]:


#fit prophet and predict

from fbprophet import Prophet

pro = Prophet(daily_seasonality=False, yearly_seasonality=False)
pro.fit(train_US_pr)
pred = pro.make_future_dataframe(periods = len(test_US_pr))
prophet_pred = pro.predict(pred)


# In[72]:


#print columns that we will use

prophet_pred[['ds','yhat','yhat_lower','yhat_upper']].tail()


# In[73]:


#convert the numbers back from log scale
prophet_pred['Prophet_yhat'] =np.exp(prophet_pred['yhat'])
prophet_pred['Prophet_yhat_lower']=np.exp(prophet_pred['yhat_lower'])
prophet_pred['Prophet_yhat_upper']=np.exp(prophet_pred['yhat_upper'])

prophet_pred.head(-5)


# In[74]:


#Convert the numbers back from log scale

prophet_forecast = prophet_pred.loc[:,('ds','yhat','yhat_lower','yhat_upper')]
prophet_forecast['yhat'] =np.exp(prophet_forecast['yhat'])
prophet_forecast['yhat_lower']=np.exp(prophet_forecast['yhat_lower'])
prophet_forecast['yhat_upper']=np.exp(prophet_forecast['yhat_upper'])


# In[75]:


prophet_forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()


# In[76]:


#set ds as index
prophet_pred = prophet_pred.set_index('ds')


# In[77]:


#add yhat, yhat_lower, yhat_upper to test_US and change column names

test_US = test_US.merge(prophet_pred[['Prophet_yhat','Prophet_yhat_lower','Prophet_yhat_upper']],
                        how ='left',left_index=True, right_index=True)


# In[78]:


#set ds as index
prophet_pred = prophet_pred.reset_index('ds')


# In[79]:


#plot prophet yhat, yhat_lower, and yhat_upper

pro.plot(prophet_pred)
plt.savefig(r'/home/songy4/Documents/prophet_log_range.png')


# In[80]:


pro.plot_components(prophet_pred)
plt.savefig(r'/home/songy4/Documents/prophet_components.png')


# In[81]:


#calculate rmse, mse errors and mean value

prophet_rmse_error = rmse(test_US['Total_case'], test_US['Prophet_yhat'])
prophet_mse_error = prophet_rmse_error **2
mean_value = df_US['Total_case'].mean()

print(f'RMSE Error: {prophet_rmse_error} \n  MSE Error: {prophet_mse_error} \n Mean: {mean_value}')


# In[83]:


#accuracy metrics: MAPE, correlation, min-max error

prophet_Mean_Forecast_Errors = np.mean(test_US.Total_case-test_US.Prophet_yhat)
prophet_MAPE = np.mean(np.abs((prophet_Mean_Forecast_Errors)/test_US.Total_case))*100
prophet_Corr = np.corrcoef(test_US.Prophet_yhat, test_US.Total_case)[0,1]
prophet_Min_Max_Error = 1 - np.mean(prophet_pred.Prophet_yhat_lower/prophet_pred.Prophet_yhat_upper)

print(f'Mean Forecast_error: {prophet_Mean_Forecast_Errors} \n MAPE: {prophet_MAPE} \n Correlation: {prophet_Corr} \n Min Max Error: {prophet_Min_Max_Error}')


# In[156]:


#plot prophet forecast with test dataset

fig_10 = plt.figure(figsize = (25, 12))
ax = sns.lineplot(x=test_US.index, y=test_US['Total_case'], label='US Total Case')
sns.lineplot(x=test_US.index, y=test_US['Prophet_yhat'], label='Prophet Predictions')
ax.legend()


# In[157]:


#Save a plot
fig_10.savefig(r'/home/songy4/Documents/prophet.png')


# In[160]:


#plot all forecasts with test dataset

fig_11 = plt.figure(figsize = (25, 12))
ax = sns.lineplot(x=test_US.index, y=test_US['Total_case'], label='US Total Case')
sns.lineplot(x=test_US.index, y=test_US['ARIMA_Predictions'], label='ARIMA Predictions')
sns.lineplot(x=test_US.index, y=test_US['LSTM_Predictions'], label='LSTM Predictions')
sns.lineplot(x=test_US.index, y=test_US['Prophet_yhat'], label='Prophet Predictions')
ax.legend()


# In[161]:


#Save a plot

fig_11.savefig(r'/home/songy4/Documents/all_forecasts.png')

