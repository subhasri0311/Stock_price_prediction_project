Stock_price.py
#Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
# Load the cleaned data
file_path = 'TVSMOTORS_cleaned.xlsx'
df = pd.read_excel(file_path)
print(df.head())

# Display the first few rows and check for missing values
print(df.head())
print(df.isnull().sum())

# Handle missing values
df.fillna(df.median(), inplace=True)

# Ensure correct data types
df['Date'] = pd.to_datetime(df['Date'])

# Example feature engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsMonthStart'] = df['Date'].dt.is_month_start
df['IsMonthEnd'] = df['Date'].dt.is_month_end

# Drop the original Date column if not needed
df.drop(columns=['Date'], inplace=True)
# Check for missing values
print(df.isnull().sum())
# Features and target variable
X = df.drop(columns=['Close'])  # Assuming 'Close' is the target
y = df['Close']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

# Evaluate Random Forest model
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_mae = mean_absolute_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)
rf_mape = mean_absolute_percentage_error(y_test, rf_y_pred)

print(f'Random Forest - Mean Squared Error: {rf_mse}')
print(f'Random Forest - Mean Absolute Error: {rf_mae}')
print(f'Random Forest - R^2 Score: {rf_r2}')
print(f'Random Forest - Mean Absolute Percentage Error: {rf_mape}')
# Initialize and train Support Vector Machine (SVM) model
svm_model = SVR()
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)

# Evaluate SVM model
svm_mse = mean_squared_error(y_test, svm_y_pred)
svm_mae = mean_absolute_error(y_test, svm_y_pred)
svm_r2 = r2_score(y_test, svm_y_pred)
svm_mape = mean_absolute_percentage_error(y_test, svm_y_pred)

print(f'SVM - Mean Squared Error: {svm_mse}')
print(f'SVM - Mean Absolute Error: {svm_mae}')
print(f'SVM - R^2 Score: {svm_r2}')
print(f'SVM - Mean Absolute Percentage Error: {svm_mape}')

# Initialize and train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)

# Evaluate Linear Regression model
lr_mse = mean_squared_error(y_test, lr_y_pred)
lr_mae = mean_absolute_error(y_test, lr_y_pred)
lr_r2 = r2_score(y_test, lr_y_pred)
lr_mape = mean_absolute_percentage_error(y_test, lr_y_pred)

print(f'Linear Regression - Mean Squared Error: {lr_mse}')
print(f'Linear Regression - Mean Absolute Error: {lr_mae}')
print(f'Linear Regression - R^2 Score: {lr_r2}')
print(f'Linear Regression - Mean Absolute Percentage Error: {lr_mape}')

# Compare the models
print("\nModel Comparison:")
print(f"Random Forest - MSE: {rf_mse}, MAE: {rf_mae}, R^2: {rf_r2}, MAPE: {rf_mape}")
print(f"SVM - MSE: {svm_mse}, MAE: {svm_mae}, R^2: {svm_r2}, MAPE: {svm_mape}")
print(f"Linear Regression - MSE: {lr_mse}, MAE: {lr_mae}, R^2: {lr_r2}, MAPE: {lr_mape}")
# Determine the best model based on metrics
best_model = min([(rf_mse, 'Random Forest', rf_mae, rf_r2, rf_mape), (svm_mse, 'SVM', svm_mae, svm_r2, svm_mape), (lr_mse, 'Linear Regression', lr_mae, lr_r2, lr_mape)],  key=lambda x: (x[0], x[2])) 
# prioritize low MSE and MAE
print(f"\nBest Model based on MSE and MAE: {best_model[1]}")

#plotting among different models
plt.figure(figsize=(14, 8)) 
plt.plot(df.index[-len(y_test):], y_test, label='Actual') 
plt.plot(df.index[-len(y_test):], lr_y_pred, label='Linear Regression') 
plt.plot(df.index[-len(y_test):], svm_y_pred, label='SVM') 
plt.plot(df.index[-len(y_test):], rf_y_pred, label='Random Forest') 
plt.title('Stock Price Prediction Comparison') 
plt.xlabel('Date') 
plt.ylabel('Stock Price') 
plt.legend() 
plt.show() 

#Hyperparameter Tuning
# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters
print(f'Best parameters: {grid_search.best_params_}')

# Best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the best model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (after tuning): {mse}')

#Exploratory data analysis
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('TVS Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()

#distribution plot
features = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sns.distplot(df[col])
plt.show()

#Box plot
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sns.boxplot(df[col])
plt.show()

#Bar graph
data_grouped = df.groupby('Year').mean(numeric_only=True)
plt.subplots(figsize=(20,10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    plt.subplot(2,2,i+1)
    data_grouped[col].plot.bar()
plt.show()

#adding some columns to train the model
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
print(df.head())

#checking whether the target is balanced or not using piechart
plt.pie(df['target'].value_counts().values,labels=[0, 1], autopct='%1.1f%%')
plt.show()

#heatmap
plt.figure(figsize=(10, 10))
numeric_dataset=df.select_dtypes(include=["int64","float64"])
sns.heatmap(numeric_dataset.corr() > 0.9, annot=True, cbar=False,cmap='Set2')
plt.show()

# Plot feature importances
feature_importances = best_model.feature_importances_
features = X.columns
importances_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importances_df)
plt.title('Feature Importances')
plt.show()

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Learning curve
train_sizes, train_scores, test_scores = learning_curve(best_model, X, y, cv=3, n_jobs=-1)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Train Score')
plt.plot(train_sizes, test_scores.mean(axis=1), label='Validation Score')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend()
plt.show()

# Rolling statistics
df['Rolling_Mean'] = df['Close'].rolling(window=30).mean()
df['Rolling_Std'] = df['Close'].rolling(window=30).std()
plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Close Price')
plt.plot(df['Rolling_Mean'], label='30-Day Rolling Mean', linestyle='--')
plt.plot(df['Rolling_Std'], label='30-Day Rolling Std Dev', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Rolling Mean and Std Dev of Close Price')
plt.legend()
plt.show()

# Box plot of daily returns
plt.figure(figsize=(14, 7))
sns.boxplot(x=df['Daily_Return'])
plt.xlabel('Daily Return')
plt.title('Box Plot of Daily Returns')
plt.show()

# Cumulative returns
df['Cumulative_Return'] = (df['Close'].pct_change() + 1).cumprod()
plt.figure(figsize=(14, 7))
plt.plot(df['Cumulative_Return'])
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Cumulative Returns Over Time')
plt.show()

#Next Day Prediction`
# Assuming df is already defined and contains a 'Close' column
# Create lagged feature for next-day prediction
df['Next_Day_Close'] = df['Close'].shift(-1)
df.dropna(inplace=True)  # Drop rows with NaN values resulting from shifting

# Update features and target
X = df.drop(columns=['Close', 'Next_Day_Close'])
y = df['Next_Day_Close']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f'Mean Squared Error (Next-Day Prediction): {mse}')
print(f'Mean Absolute Error (Next-Day Prediction): {mae}')
print(f'R^2 Score (Next-Day Prediction): {r2}')
print(f'Mean Absolute Percentage Error (Next-Day Prediction): {mape}%')
