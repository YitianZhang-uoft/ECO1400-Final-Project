# Import required packages
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller

%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.vector_ar.vecm import *
import pandas

# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.vector_ar.vecm import VECM

# Import performance metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

import math
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("select_m_data_log.csv",parse_dates=["Date"],index_col=[0])

# Split the data into training and testing data
n_obs = 72
X_train, X_test = df[0:-n_obs], df[-n_obs:]

# Check size
print(X_train.shape)
print(X_test.shape)

# Johansen Cointegration Test
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

cointegration_test(X_train)

# Lag order selection
X_train = X_train.drop(['CPIAUCSL', 'MVIXADCLS'], axis = 1)
X_test = X_test.drop(['CPIAUCSL', 'MVIXADCLS'], axis = 1)
lag_order = select_order(data = X_train, maxlags=10, deterministic="ci", seasons=4)
lag_order.summary()

rank_test = select_coint_rank(X_train, 0, 5, method="trace", signif=0.05)
rank_test.rank

# Parameter Estimation
model = VECM(X_train, deterministic = "ci", seasons = 4, k_ar_diff = 2, coint_rank = 2)

vecm_res = model.fit()

vecm_res.summary()

# Forecasts
forecast = vecm_res.predict(72, 0.05)

pred = (pd.DataFrame(forecast[0], index = X_test.index, columns = X_test.columns + '_pred'))

# Plot actual value versus predicted value
plt.figure(figsize = (12, 5))
plt.xlabel('Date')

ax1 = X_test.MCOILWTICO.plot(color = 'blue', grid = True, label = 'Actual Oil Log-Price')
ax2 = pred.MCOILWTICO_pred.plot(color = 'red', grid = True, label = 'Predicted Oil Log-Price')

ax1.legend(loc = 1)
ax2.legend(loc = 2)
plt.title('Predicted V.S. Actual Oil Log-Price')
plt.show()

# Calculate performance metrics

#Calculate mean absolute error
mae = mean_absolute_error(X_test['MCOILWTICO'], pred['MCOILWTICO_pred'])
print('MAE: %f' % mae)

#Calculate mean squared error
rmse = math.sqrt(mean_squared_error(X_test['MCOILWTICO'], pred['MCOILWTICO_pred']))
print('RMSE: %f' % rmse)

# Calculate mean absolute percentage error
mape = np.mean(np.abs((X_test['MCOILWTICO'] - pred['MCOILWTICO_pred'])/X_test['MCOILWTICO']))
print('MAPE: %f' % mape)