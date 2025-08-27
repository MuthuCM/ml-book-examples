# Example 15.2
# PCA for Portfolio Management
# Step 1 : Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Step 2 : Read dataset
share_prices_df = pd.read_csv('dow_share_prices1.csv',index_col=0)
# share_prices_df.head()
# Do Preprocessing
missing_fractions = share_prices_df.isnull().mean().sort_values(ascending=False)
missing_fractions.head(5)

drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))

share_prices_df.drop(labels=drop_list, axis=1, inplace=True)
share_prices_df.shape
# Fill the missing values with the last value available in the dataset.
share_prices_df=share_prices_df.fillna(method='ffill')

# Drop the rows containing NA
share_prices_df= share_prices_df.dropna(axis=0)
# Fill na with 0
#dataset.fillna('0')
share_prices_df.head(3)

# Step 3 : Calculate Daily Log Returns (%)
# Calculate Daily Linear Returns (%)
datareturns = share_prices_df.pct_change(1)

#Remove Outliers beyong 3 standard deviation
datareturns= datareturns[datareturns.apply(lambda x :(x-x.mean()).abs()<(3*x.std()) ).all(1)]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(datareturns)
rescaledDF = pd.DataFrame(scaler.fit_transform(datareturns),columns =   
                                                                             datareturns.columns, index = datareturns.index)
# summarize transformed data
datareturns.dropna(how='any', inplace=True)
rescaledDF.dropna(how='any', inplace=True)
rescaledDF.head(3)

# Step 4 : Visualize Log Returns for the Dow Jones Industrial Average
plt.figure(figsize=(16, 5))
plt.title("AXP Return")
plt.ylabel("Return")
rescaledDF.AXP.plot()
plt.grid(True);
plt.legend()
plt.show()

# Step 5 : Divide the dataset into training and testing sets
percentage = int(len(rescaledDF) * 0.99)
X_train = rescaledDF[:percentage]
X_test = rescaledDF[percentage:]

X_train_raw = datareturns[:percentage]
X_test_raw = datareturns[percentage:]


stock_tickers = rescaledDF.columns.values
n_tickers = len(stock_tickers)

# Step 6 : Extract Principal Components
pca = PCA()
PrincipalComponent=pca.fit(X_train)
pca.components_[1]

# Step 7 : Find Principal Components (PC) weights 
weights = pd.DataFrame()
for i in range(len(pca.components_)):
  weights["weights_{}".format(i)] = pca.components_[i] / sum(pca.components_[i])
weights = weights.values.T

# Step 8 : Visualize First Two Components
NumPortfolios=2

topPortfolios = pd.DataFrame(pca.components_[:NumPortfolios], columns=share_prices_df.columns)
PCA_portfolios = topPortfolios.div(topPortfolios.sum(1), axis=0)
PCA_portfolios.index = [f'Portfolio {i}' for i in range( NumPortfolios)]
np.sqrt(pca.explained_variance_)
PCA_portfolios.T.plot.bar(subplots=True, layout=(int(NumPortfolios),1)
                               , figsize=(14,10), legend=False, sharey=True, ylim= (-1,1))
