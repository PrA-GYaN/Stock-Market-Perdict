from types import DynamicClassAttribute
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import mod
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout  , LSTM
from tensorflow import keras

#Load Data
company = "FB"
start = dt.datetime(2012,1,1)
end = dt.datetime(2021,1,1)

#Reading data from Yahoo Finance
data = web.DataReader(company , 'yahoo' , start , end)

#Prepare Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data= scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_day = 60 
x_train = []
y_train = []

for x in range(prediction_day , len(scaled_data)):
    x_train.append(scaled_data[x-prediction_day:x , 0])
    y_train.append(scaled_data[x,0])

x_train, y_train = np.array(x_train) , np.array(y_train)
x_train = np.reshape(x_train , (x_train.shape[0] , x_train.shape[1] , 1))

#Build  Model
model = Sequential()

model.add(LSTM(units = 62 ,return_sequences = True , input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 62 ,return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 62))
model.add(Dropout(0.2))
model.add(Dense(units=1))

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile( loss = 'mean_squared_error', optimizer=opt)
model.fit(x_train, y_train , epochs = 38 , batch_size=32)

#Load Data

test_start = dt.datetime(2021,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(company , 'yahoo' , test_start , test_end)
actual_price = test_data['Close'].values

total_dataset = pd.concat((data['Close'] , test_data['Close']) , axis=0)

model_input = total_dataset[len(total_dataset) - len(test_data) - prediction_day:].values
model_input = model_input.reshape(-1 , 1)
model_input = scaler.transform(model_input)

#prediction in test data

x_test = []
for x in range(prediction_day , len(model_input)):
    x_test.append(model_input[x - prediction_day:x , 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test , (x_test.shape[0] , x_test.shape[1] , 1))

prediction_price = model.predict(x_test)
prediction_price = scaler.inverse_transform(prediction_price)

#plot prediction
plt.plot(actual_price ,color= "black", label = f"actual {company} Price")
plt.plot(prediction_price, color = "green" , label = f"predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()

#prediction 
real_data = [model_input[len(model_input) +1 - prediction_day:len(model_input + 1) , 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data , (real_data.shape[0],real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)

print(f"Prediction : {prediction}")
