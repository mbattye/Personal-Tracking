#Import Modules
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib.dates as mdat
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#import seaborn as sns
#import statsmodels.api as sm
#from statsmodels.sandbox.regression.predstd import wls_prediction_std
#from sklearn import linear_model

file='/Users/michaelbattye/Desktop/Python/Data Sets/moment.json'

array = [['Date', 'Day', 'Pick Up Count', 'Minutes']]

#Dictionary from JSON
dict = pd.read_json(file)["days"]

for row in dict:
    array.append([dt.strptime(row["date"][:10], '%Y-%m-%d').strftime('%d/%m/%y'), dt.strptime(row["date"][:10], '%Y-%m-%d').date().weekday(), row["pickupCount"], row["minuteCount"]])

#Dataframe and sub-series
df = pd.DataFrame(array)
Date=df.iloc[:,0]
Days=df.iloc[:,1]
PUC=df.iloc[:,2]
Minutes=df.iloc[:,3]

#Initiate Weekday Minute Arrays
monmins = []
tuemins = []
wedmins = []
thumins = []
frimins = []
satmins = []
sunmins = []
#Initiate Weekday Pick-Up Arrays
monpu = []
tuepu = []
wedpu = []
thupu = []
fripu = []
satpu = []
sunpu = []

#Define Weekday Minute Function
def dayminutes(day, m):
    count=0
    for d in day:
        if d == 0:
            monmins.append(m[count])
        if d == 1:
            tuemins.append(m[count])
        if d == 2:
            wedmins.append(m[count])
        if d == 3:
            thumins.append(m[count])
        if d == 4:
            frimins.append(m[count])
        if d == 5:
            satmins.append(m[count])
        if d == 6:
            sunmins.append(m[count])
        count +=1

#Define Weekday Pick-Up Function
def daypu(day, m):
    count=0
    for d in day:
        if d == 0:
            monpu.append(m[count])
        if d == 1:
            tuepu.append(m[count])
        if d == 2:
            wedpu.append(m[count])
        if d == 3:
            thupu.append(m[count])
        if d == 4:
            fripu.append(m[count])
        if d == 5:
            satpu.append(m[count])
        if d == 6:
            sunpu.append(m[count])
        count +=1

#Apply Weekday Minute Function to Days & Minutes
dayminutes(Days, Minutes)

#Apply Weekday Pick-Up Function to Days & Pick-Ups
daypu(Days, PUC)

#Convert Day-Minute Lists to Arrays
m = np.asarray(monmins)
tu = np.asarray(tuemins)
w = np.array(wedmins)
th = np.asarray(thumins)
f = np.asarray(frimins)
sa = np.asarray(satmins)
su = np.asarray(sunmins)

#Convert Day-PickUp Lists to Arrays
mpu = np.asarray(monpu)
tupu = np.asarray(tuepu)
wpu = np.array(wedpu)
thpu = np.asarray(thupu)
fpu = np.asarray(fripu)
sapu = np.asarray(satpu)
supu = np.asarray(sunpu)

#Print overall stats
print("Average Minutes Spent on Screen:", round(Minutes.iloc[1:].mean(), 2))
print("Standard Deviation of Minutes Spent on Screen:", round(Minutes.iloc[1:].std(), 2))
print("Average Pick-Ups:", round(PUC.iloc[1:].mean(), 2))
#Day of the week stats
#Minutes Spent on Screen
print("Average Minutes Spent on Screen on Mondays:", round(m[1:].mean(), 2))
print("Average Minutes Spent on Screen on Tuesdays:", round(tu[1:].mean(), 2))
print("Average Minutes Spent on Screen on Wednesdays:", round(w[1:].mean(), 2))
print("Average Minutes Spent on Screen on Thurdays:", round(th[1:].mean(), 2))
print("Average Minutes Spent on Screen on Fridays:", round(f[1:].mean(), 2))
print("Average Minutes Spent on Screen on Saturdays:", round(sa[1:].mean(), 2))
print("Average Minutes Spent on Screen on Sundays:", round(su[1:].mean(), 2))
#Pick-Ups
print("Average Pick-Ups on Mondays:", round(mpu[1:].mean(), 2))
print("Average Pick-Ups on Tuesdays:", round(tupu[1:].mean(), 2))
print("Average Pick-Ups on Wednesdays:", round(wpu[1:].mean(), 2))
print("Average Pick-Ups on Thurdays:", round(thpu[1:].mean(), 2))
print("Average Pick-Ups on Fridays:", round(fpu[1:].mean(), 2))
print("Average Pick-Ups on Saturdays:", round(sapu[1:].mean(), 2))
print("Average Pick-Ups on Sundays:", round(supu[1:].mean(), 2))

fig, ax = plt.subplots()
ax.plot_date(Date, Minutes)
#x-axis
startx, endx = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(startx, endx, 50))
plt.show()
