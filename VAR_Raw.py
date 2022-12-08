# Import required packages
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller

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
df = pd.read_csv("select_m_data_bsw.csv",parse_dates=["Date"],index_col=[0])

# Check data set
print(df.head())
print('\n')
print(df.tail())
print('\n')
df.describe()

# plots the autocorrelation plots at 75 lags
for i in df:
 plot_acf(df[i], lags = 75)
 plt.title('ACF for %s' % i) 
 plt.show()

# Split the data into training and testing data
n_obs = 72
X_train, X_test = df[0:-n_obs], df[-n_obs:]

# Check size
print(X_train.shape)
print(X_test.shape)

# Stationary check
def augmented_dickey_fuller_statistics(time_series):
    result = adfuller(time_series.values)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

# ADF Test on each column
def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")   

for name, column in X_train.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')
    
# Applying 1st order difference
X_train_diff =(X_train).diff().dropna()
X_train_diff.describe()

X_train_diff.plot(figsize = (10,6), linewidth = 5, fontsize = 14)

# ADF Test - transformed series
for name, column in X_train_diff.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')
    
# Granger Calsuality Test
from statsmodels.tsa.stattools import grangercausalitytests
maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

grangers_causation_matrix(X_train_diff, variables = X_train_diff.columns) 

X_train_diff = X_train_diff.drop(['CPILFESL', 'CUSR0000SETA02', 'UNRATE', 'MCOILBRENTEU', 'MSACSR', 'UNRATE'], axis = 1)
X_test = X_test.drop(['CPILFESL', 'CUSR0000SETA02', 'UNRATE', 'MCOILBRENTEU', 'MSACSR', 'UNRATE'], axis = 1)
X_train = X_train.drop(['CPILFESL', 'CUSR0000SETA02', 'UNRATE', 'MCOILBRENTEU', 'MSACSR', 'UNRATE'], axis = 1)

#Initiate VAR model
model = VAR(endog=X_train_diff)
res = model.select_order(10)
res.summary()

#Fit to a VAR model
model_fit = model.fit(2)
#Print a summary of the model results
model_fit.summary()

# Durbin Watson Test
from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fit.resid)

def adjust(val, length= 6): return str(val).ljust(length)

for col, val in zip(df.columns, out):
    print(adjust(col), ':', round(val, 2))
    
# Get the lag order
lag_order = model_fit.k_ar
print(lag_order)

# Input data for forecasting
input_data = X_train_diff.values[-lag_order:]
print(input_data)

# forecasting
pred = model_fit.forecast(y = input_data, steps = n_obs)
pred = (pd.DataFrame(pred, index = X_test.index, columns = X_test.columns + '_pred'))
print(pred)

# inverting transformation
def invert_transformation(df_train, df_forecast):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_pred'].cumsum()
    return df_fc

df_results = invert_transformation(X_train, pred)   
df_results.loc[:, ['MCOILWTICO_forecast']]
#df_results.loc[:, ['DCOILWTICO_forecast', 'DCOILBRENTEU_forecast', 'DEXCAUS_forecast', 'DEXCHUS_forecast', 'DEXJPUS_forecast', 'DEXUSEU_forecast', 'DEXUSUK_forecast', 'DTB3_forecast', 'T10Y2Y_forecast', 'T5YIE_forecast', 'T5YIFR_forecast', 'NASDAQCOM_forecast', 'SP500_forecast', 'WILL5000INDFC_forecast', 'VIXOPN_forecast', 'VIXADCLS_forecast', 'GCCMXCLS_forecast']]

# Plot actual value versus predicted value
plt.figure(figsize = (12, 5))
plt.xlabel('Date')

ax1 = X_test.MCOILWTICO.plot(color = 'blue', grid = True, label = 'Actual Oil log_Price')
ax2 = df_results.MCOILWTICO_forecast.plot(color = 'red', grid = True, label = 'Predicted Oil Log-Price')

ax1.legend(loc = 1)
ax2.legend(loc = 2)
plt.title('Predicted V.S. Actual Oil Log-Price')
plt.show()

# Calculate performance metrics

#Calculate mean absolute error
mae = mean_absolute_error(X_test['MCOILWTICO'], df_results['MCOILWTICO_forecast'])
print('MAE: %f' % mae)

#Calculate mean squared error
rmse = math.sqrt(mean_squared_error(X_test['MCOILWTICO'], df_results['MCOILWTICO_forecast']))
print('RMSE: %f' % rmse)

# Calculate mean absolute percentage error
mape = np.mean(np.abs((X_test['MCOILWTICO'] - pred['MCOILWTICO_pred'])/X_test['MCOILWTICO']))
print('MAPE: %f' % mape)