"""
Author: Devin Sigler

Objective:

"""
# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Create Train and Test dataframe
dfTrain = pd.read_csv('train.csv')
dfTest  = pd.read_csv('test.csv')

# Print a plot of the sale price data to show the normal curve
data   = dfTrain['SalePrice']
result = sns.displot(data)
plt.show()

# Find the Skew and Kurtosis (To imitate a normal curve, you want a skew of 0 and a Kurtosis of 3)
print("Skewness: %f" % dfTrain['SalePrice'].skew())
print("Kurtosis: %f" % dfTrain['SalePrice'].kurt())
# This data has a positive skew (leans left), and a kurtosis of 6 (peaks are higher) but it is enough to be modelled accurately

# Show a scatterplot of the SalePrice and GrLivArea (size of property)
var  = 'GrLivArea'
data = pd.concat([dfTrain['SalePrice'], dfTrain[var]], axis = 1)
data.plot.scatter(x = var, y = 'SalePrice', ylim = (0, 800000))
plt.show()
# As the size of the houses increase, you can see an increase in price (Correlation does not imply causation, but there seems to be a linear relationship)

# Show another scatterplot of the SalePrice and TotalBsmtSF (Basement square feet, another size indicator)
var  = 'TotalBsmtSF'
data = pd.concat([dfTrain['SalePrice'], dfTrain[var]], axis = 1)
data.plot.scatter(x = var, y = 'SalePrice', ylim = (0, 800000))
plt.show()
# This plot shows a line along the y axis (homes without a basement), but the plot almost seems to imply an exponential relationship

# OverallQual is a variable of 1-10 like a rating system.  I don't know how this was judged, but I wanted to group them with a box plot to show the distributions within each quality rating
var   = 'OverallQual'
data  = pd.concat([dfTrain['SalePrice'], dfTrain[var]], axis = 1)
f, ax = plt.subplots(figsize = (8, 6))
fig   = sns.boxplot(x = var, y = 'SalePrice', data=data)
fig.axis(ymin = 0, ymax = 800000)
plt.show()
# As I believed, the higher the OverallQuality, the higher the price.  This seems to increase exponentially with each level of OverallQual

# Now as I did with OverallQual, I'm going to do another box plot with YearBuilt as the x - axis
var   = 'YearBuilt'
data  = pd.concat([dfTrain['SalePrice'], dfTrain[var]], axis = 1)
f, ax = plt.subplots(figsize = (8, 8))
fig   = sns.boxplot(x = var, y = 'SalePrice', data=data)
fig.axis(ymin = 0, ymax = 800000)
plt.xticks(rotation = 90)
plt.show()
# I anticipated more of a slope to the relationship between Year and Price, but their is still a linear relatinship that seems to indicate the price goes higher with newer homes

# A Correlation Matrix would show that very quickly what variables have strong relationships with each other
corMat = dfTrain.corr()
f, ax  = plt.subplots(figsize = (9, 12))
sns.heatmap(corMat, vmax = .8, square = True)
plt.show()

# Now I wanted to show a SalePrice Correlation Matrix after looking at their individual relationships with variables that stuck out
i    = 10
cols = corMat.nlargest(i, 'SalePrice')['SalePrice'].index
cM   = np.corrcoef(dfTrain[cols].values.T)
sns.set(font_scale = 1.5)
hm   = sns.heatmap(cM, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size' : 10}, yticklabels = cols.values, xticklabels = cols.values)
plt.show()

# This is a graph I may have gone overboard with, but its a scatterplot of each block of the previous Correlation Matrix
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(dfTrain[cols], height = 2.5)
plt.show()

# Now I wanted to take a look at what data is missing so I know what to ignore and throw out
total        = dfTrain.isnull().sum().sort_values(ascending=False)
percent      = (dfTrain.isnull().sum()/dfTrain.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))

# From this, I can throw out PoolQC, MiscFeature, Alley, Fence, FireplaceQu, and LotFrontage because they are missing more than 15% each
dfTrain = dfTrain.drop((missing_data[missing_data['Total'] > 1]).index,1)
dfTrain = dfTrain.drop(dfTrain.loc[dfTrain['Electrical'].isnull()].index)
dfTrain.isnull().sum().max()

# Now to find the Outliers!
salePriceScaled = StandardScaler().fit_transform(dfTrain['SalePrice'][:,np.newaxis])
low_range       = salePriceScaled[salePriceScaled[:,0].argsort()][:10]
high_range      = salePriceScaled[salePriceScaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(f'{low_range}\n')
print('outer range (high) of the distribution:')
print(f'{high_range}\n')

# And remove them
dfTrain.sort_values(by = 'GrLivArea', ascending = False)[:2]
dfTrain = dfTrain.drop(dfTrain[dfTrain['Id'] == 1299].index)
dfTrain = dfTrain.drop(dfTrain[dfTrain['Id'] == 524].index)

# Now lets look at living area without those outliers
var = 'GrLivArea'
data = pd.concat([dfTrain['SalePrice'], dfTrain[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
plt.show()

var = 'TotalBsmtSF'
data = pd.concat([dfTrain['SalePrice'], dfTrain[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
plt.show()

# With the modified data, lets replot the sale price histogram with the normal graph
sns.displot(dfTrain['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(dfTrain['SalePrice'], plot=plt)
plt.show()

# Since the probability plot was showing to be exponential and not linear, I wanted to try a log transformation to see what that did to the data set
dfTrain['SalePrice'] = np.log(dfTrain['SalePrice'])
sns.displot(dfTrain['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(dfTrain['SalePrice'], plot=plt)
plt.show()

# That transformation looked great, so lets do the same on GrLivArea, showing the before and after
sns.displot(dfTrain['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(dfTrain['GrLivArea'], plot=plt)
plt.show()

dfTrain['GrLivArea'] = np.log(dfTrain['GrLivArea'])
sns.displot(dfTrain['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(dfTrain['GrLivArea'], plot=plt)
plt.show()

# TotalBsmtSF before any transformations
sns.displot(dfTrain['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(dfTrain['TotalBsmtSF'], plot=plt)
plt.show()