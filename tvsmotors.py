import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# Verify the file path
file_path = 'TVSMOTORS.csv'

# Check if the file exists
import os
if not os.path.exists(file_path):
print(f"File not found: {file_path}")
else:
           # Load data from a CSV file
df = pd.read_csv(file_path)
# Explore the data
 print(df.head())
print("\n")
print(df.info())
print("\n")
print(df.describe())

# Handle missing data
print(df.isnull().sum())
df.dropna(inplace=True)
# Alternatively, df.fillna(df.mean(), inplace=True)
# Remove unnecessary columns
columns_to_drop = [ 'Adj Close']  # Replace with actual column names to drop
df.drop(columns=columns_to_drop, axis=1, inplace=True)

# Convert data types
df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)
df['Close'] = df['Close'].astype(float)
# Handle duplicates
df.drop_duplicates(inplace=True)

# Feature engineering
df['Moving_Avg_5'] = df['Close'].rolling(window=5).mean()
df['Moving_Avg_10'] = df['Close'].rolling(window=10).mean()
df['Daily_Return'] = df['Close'].pct_change()

# Normalize/scale data
scaler = MinMaxScaler()
df['Close_Scaled'] = scaler.fit_transform(df[['Close']])
df.fillna(df.median(), inplace=True)
# Save cleaned data to a new Excel file
df.to_excel('TVSMOTORS_cleaned.xlsx', index=False)

