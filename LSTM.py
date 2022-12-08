# Import required packages
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

import math
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("select_m_data_log.csv",parse_dates=["Date"],index_col=[0])

# Check dataframe
print(df.shape)
print(df.head())
print(df.tail())

# Set cut off for splitting into training and testing set (70% training and 30% testing)
test_split = round(len(df)*0.3)
test_split

# Split into training and testing test and check
df_for_training=df[:-test_split]
df_for_testing=df[-test_split:]
print(df_for_training.shape)
print(df_for_testing.shape)

# Notice that the data range is very high, and they are not scaled in a same range, so to avoid prediction errors, we scale the data first
scaler = MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled=scaler.transform(df_for_testing)
df_for_training_scaled

# Split training data into X variable (independent) and Y variable (dependent) and set the timestep for forecasting
timestep = 5

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX), np.array(dataY)
trainX,trainY=createXY(df_for_training_scaled, timestep)
testX,testY=createXY(df_for_testing_scaled, timestep)

trainX,trainY=createXY(df_for_training_scaled,timestep)
testX,testY=createXY(df_for_testing_scaled,timestep)

# Check the X and Y variable shape
print("trainX Shape-- ",trainX.shape)
print("trainY Shape-- ",trainY.shape)

print("testX Shape-- ",testX.shape)
print("testY Shape-- ",testY.shape)

# Construct the baseline model by hyperparameter tuning to pick the optimal parameter
def build_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(50, return_sequences=True, input_shape=(timestep,7)))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))
    
    grid_model.compile(loss = 'mse',optimizer = optimizer)
    return grid_model

grid_model = KerasRegressor(build_fn = build_model, verbose = 1, validation_data = (testX, testY))
parameters = {'batch_size' : [8, 16, 32],
              'epochs' : [8, 16, 32],
              'optimizer' : ['adam','Adadelta'] }

grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 2)

grid_search = grid_search.fit(trainX,trainY)

# Show the selected optimal parameter
grid_search.best_params_

# Store the model with optimal parameter
LSTM_model = grid_search.best_estimator_.model

# Get the prediction value
prediction = LSTM_model.predict(testX)
print("prediction\n", prediction)
print("\nPrediction Shape-",prediction.shape)

# Inverse transform the predicted value
prediction_copies_array = np.repeat(prediction,7, axis=-1)

prediction_copies_array.shape

pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),7)))[:,0]

# Inverse transform the test dependent variable
original_copies_array = np.repeat(testY,7, axis=-1)
original=scaler.inverse_transform(np.reshape(original_copies_array,(len(testY),7)))[:,0]

plt.plot(original, color = 'red', label = 'Real Oil Log-Price')
plt.plot(pred, color = 'blue', label = 'Predicted Oil Log-Price')
plt.title('Oil Log-Price Prediction')
plt.xlabel('Time')
plt.ylabel('Oil Log-Price')
plt.legend()
plt.show()

# Caldulate Performance Metrics
import sklearn

MAE = sklearn.metrics.mean_absolute_error(original, pred)
MAPE = sklearn.metrics.mean_absolute_percentage_error(original, pred)
RMSE = math.sqrt(np.square(np.subtract(original, pred)).mean())
print(RMSE)
print(MAE)
print(MAPE)