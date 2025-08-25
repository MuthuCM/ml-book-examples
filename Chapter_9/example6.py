# Example 9.6
# Load Packages
# Example 9.6
# Load Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score

# Load Data
data = pd.read_csv('crop_yield.csv')

# Identify categorical columns for encoding
categorical_columns = ['Crop']
label_encoders = {}
for col in categorical_columns:
   le = LabelEncoder()
   data[col] = le.fit_transform(data[col])
# Define Independent and Dependent Variables
target = data['Yield']
      features=data[['Crop','Area','Production','Annual_Rainfall',                               
                                              'Fertilizer', 'Pesticide' ]]       
# Split into Training and Testing Sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (features, target,  
                                     test_size=0.3, random_state = 42)

# Do Scaling of Data
scaler = standardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)
# Fit AdaBoost Regressor Model
from sklearn.ensemble import AdaBoostRegressor
ab_model = AdaBoostRegressor(random_state=42 )
ab_model.fit(X_train_scaled,y_train)

# Do prediction with AdaBoost Regressor Model
from sklearn.metrics import mean_squared_error, r2_score
y_pred_ab = ab_model.predict(X_test_scaled)
rmse_ab = np.sqrt(mean_squared_error(y_test, y_pred_ab))
r2_ab = r2_score(y_test, y_pred_ab)
print(f'AdaBoost - RMSE: {rmse_ab}, R²: {r2_ab}')
# Fit GradientBoosting Regressor Model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
gb_model = GradientBoostingRegressor(random_state=42 )
gb_model.fit(X_train_scaled,y_train)

# Do prediction with GradientBoosting Regressor Model
from sklearn.metrics import mean_squared_error, r2_score
y_pred_gb = gb_model.predict(X_test_scaled)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb = r2_score(y_test, y_pred_gb)
print(f'Gradient Boosting - RMSE: {rmse_gb}, R²: {r2_gb}')



