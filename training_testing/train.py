import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the data
data = pd.read_csv('output/train_dataset.csv')
model_save_path = 'output/xgb_model.json'

# Separate features and target
X = data.drop(columns=['slowdown'])
y = data['slowdown']

print('X.head()', X.head())
print('y.head()', y.head())

# Split the data into train/validation and test sets (80% train/validation, 20% test)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features for train/validation and test sets
scaler = StandardScaler()
X_train_val_normalized = scaler.fit_transform(X_train_val)
X_test_normalized = scaler.transform(X_test)

# Convert normalized features back to DataFrames to keep column names
X_train_val = pd.DataFrame(X_train_val_normalized, columns=X.columns)
X_test = pd.DataFrame(X_test_normalized, columns=X.columns)

# Define the XGBoost regressor with a maximum depth of 12
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', max_depth=12, random_state=42)

# Set up k-fold cross-validation
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

mse_scores = []

# Perform k-fold cross-validation on the training/validation set
for train_index, val_index in kf.split(X_train_val):
    X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
    y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]
    
    # Train the model
    xgb_model.fit(X_train, y_train)
    
    # Predict on the validation set
    y_pred = xgb_model.predict(X_val)
    
    # Calculate the Mean Squared Error for the validation set
    mse = mean_squared_error(y_val, y_pred)
    mse_scores.append(mse)

# Output the MSE for each fold and the average MSE from cross-validation
print(f"MSE for each fold (validation set): {mse_scores}")
print(f"Average MSE (validation set): {np.mean(mse_scores)}")

# Train the final model on the entire train/validation set
xgb_model.fit(X_train_val, y_train_val)

# Save the trained model to a file
xgb_model.save_model(model_save_path)

# Predict on the test set
y_test_pred = xgb_model.predict(X_test)

# Calculate the Mean Squared Error for the test set
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Test MSE: {test_mse}")

# Load the saved model (for future use)
xgb_model_loaded = xgb.XGBRegressor()
xgb_model_loaded.load_model(model_save_path)

# Make predictions with the loaded model
y_test_pred_loaded = xgb_model_loaded.predict(X_test)

# Check if the loaded model gives the same predictions
assert np.allclose(y_test_pred, y_test_pred_loaded), "The predictions from the loaded model differ!"
print("Model loaded successfully and predictions match.")