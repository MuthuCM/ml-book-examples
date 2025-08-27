# Example 15.3
# PCA for Interest Rate Prediction
# Step 1: Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
!pip install quandl

# Step 2 : Install quandl using "!pip install quandl"
# !pip install quandl
import quandl

#The API Key can be optained from Quandl website by registering.
# IMPORTANT: Ensure you have a valid Quandl API key
quandl.ApiConfig.api_key = 'xZ2YZuFov1VMXphKmYZ_'

# Step 3: Prepare dataset
bonds = ['FRED/DGS1MO',   'FRED/DGS3MO',   'FRED/DGS6MO',
           'FRED/DGS1',                'FRED/DGS2',           'FRED/DGS3',
           'FRED/DGS5',               'FRED/DGS7',           'FRED/DGS10',
           'FRED/DGS20',             'FRED/DGS30']

bond_yields_df = quandl.get(bonds)
bond_yields_df.columns = ['TRESY1mo',   'TRESY3mo',      'TRESY6mo',
                                                  'TRESY1y',       'TRESY2y',          'TRESY3y',   'TRESY5y',
                                                   'TRESY7y',      'TRESY10y',        'TRESY20y',     'TRESY30y']
df = bond_yields_df
df.head(10)

# shape
df.shape

# Statistical Measures
df.describe()

# Step 4 : Do Preproceesing
# Drop the rows containing NA values
df= df.dropna(axis=0)
df.head(3)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(df)
rescaledDF = pd.DataFrame(scaler.fit_transform(df),columns = df.columns, index = df.index)
# summarize transformed data
df.dropna(how='any', inplace=True)
rescaledDF.dropna(how='any', inplace=True)
rescaledDF.head(3)

# Step 5 : Find Principal Components
pca = PCA()
PrincipalComponent=pca.fit(rescaledDF)
NumEigenvalues=3
fig, axes = plt.subplots(ncols=2, figsize=(14,4))
pd.Series(pca.explained_variance_ratio_[:NumEigenvalues]).sort_values().plot.barh(title='Explained Variance Ratio by Top Factors',ax=axes[0]);
pd.Series(pca.explained_variance_ratio_[:NumEigenvalues]).cumsum().plot(ylim=(0,1),ax=axes[1], title='Cumulative Explained Variance');

# Step 7 : Compute Principal Components (PC) weights
weights = pd.DataFrame()
for i in range(len(pca.components_)):
  weights["weights_{}".format(i)] = pca.components_[i] / sum(pca.components_[i])
weights = weights.values.T

NumPortfolios=3
topPortfolios = pd.DataFrame(weights[:NumPortfolios], columns=df.columns)
topPortfolios.index = [f'Principal Component {i}' for i in range(1, NumPortfolios+1)]
axes = topPortfolios.T.plot.bar(subplots=True, legend=False,figsize=(14,10))
plt.subplots_adjust(hspace=0.35)
axes[0].set_ylim(0, .2);
