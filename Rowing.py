#Import Modules
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib.dates as mdat
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn import linear_model


#Dataframe of CSV
df = pd.read_csv('/Users/michaelbattye/Desktop/Python/RowingTimes.csv', dtype=str)

#x and y data from dataframe columns
rowdate = df.iloc[:,0]
rowtime = df.iloc[:,1]

#Create blank lists for datetime versions of x, y data
rowd=[]
rowt=[]

for i in rowdate:
	rowd.append(dt.strptime(i, '%d%m%y'))

for i in rowtime:
	rowt.append(dt.strptime(i, '%M:%S'))

#Datetime to Matplotlib form
x = mdat.date2num(rowd)
y = mdat.date2num(rowt)

X = sm.add_constant(x)

# Note the difference in argument order
mod_ols = sm.OLS(y, X).fit()
pred_ols = mod_ols.predict(X) # make the predictions by the model

# Print out the OLS statistics
print(mod_ols.summary())

print('Parameters: ', mod_ols.params)
print('Standard errors: ', mod_ols.bse)
print('Predicted values: ', mod_ols.predict())

#Build Confidence Intervals around Predictions
prstd, iv_l, iv_u = wls_prediction_std(mod_ols)

#Linear Model
lm = linear_model.LinearRegression()
mod_lm = lm.fit(X,y)
pred_lm = lm.predict(X)

# Print out the LM statistics
print(mod_lm)


#Figure & Axes
fig, ax = plt.subplots()
#Figure Title
ax.set_title("Rowing Progress", color='#5C596D')

#Plot the graph
ax.plot(x, y, '.', label='Data', color='#000000')
ax.plot(x, pred_ols, 'g--.', label="OLS")
ax.plot(x, iv_u, 'g--')
ax.plot(x, iv_l, 'g--')
#ax.plot(x, pred_lm, 'r--.', label="Linear Regression")

#Label Axe
plt.xlabel("Date", labelpad = 20, color='#5C596D', fontsize = 15)
plt.ylabel("Time", rotation = 0, labelpad = 30, color='#5C596D', fontsize = 15)

#Format Axes
xFmt = mdat.DateFormatter('%d %b %y')
ax.xaxis.set_major_formatter(xFmt)

yFmt = mdat.DateFormatter('%M:%S')
ax.yaxis.set_major_formatter(yFmt)

#Line Fit
#plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), label='Line Fit', color='#FF4500')


#Seaborn Styles
sns.set(style="ticks",color_codes=True)

#plt.tight_layout()
plt.legend()
plt.show()