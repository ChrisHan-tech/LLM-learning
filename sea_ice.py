import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt

# We assume a nonlinear model: y = X @ theta
# X = [Year, sin(Month_angle), cos(Month_angle), Temperature, Solar Radiation, Average Salinity]
#=============================================
# Load and preprocess data
#=============================================
os.chdir("/Users/mac/Desktop/IMDS/miniproject")
data = pd.read_excel("sea_ice_data.xlsx")

month_to_num = {
    "January":1, "Febraury":2, "March":3, "April":4, "May":5, "June":6,
    "July":7, "August":8, "September":9, "October":10, "November":11, "December":12
}
data["Month"] = data["Month"].map(month_to_num)
data['Month_angle'] = (data['Month'] - 1) * 2 * np.pi / 12  # Change month to angle
print(data.isnull().sum()) 
data = data.dropna()
print(data)

X = data[['Year', 'Month_angle', 'Temperature(C)', 'Solar Radiation(J/m2)', 'Average Salinity(PSU)']].values
Y = data["Extent (km2)"].values
data['Solar Radiation(J/m2)'] = data['Solar Radiation(J/m2)'].apply(lambda x: np.log10(x + 1e-12)) # Logarise solar radiation

#Normalize features
X_mean = np.mean(X, axis = 0)
X_std = np.std(X, axis = 0)
X_normalized = (X - X_mean) / X_std

#=============================================
# Define Loss function with L2 regularization
#=============================================
def loss(X_bias, y, theta, lambda_reg):
    m = len(y)  # Number of samples
    #predictions = np.log(1 + np.exp(X_bias @ theta))  # nonlinear predictions
    predictions = X_bias @ theta
    mse_loss = (1 / (2 * m)) * np.sum((predictions - y) ** 2)  # Mean Squared Error
    reg_loss = (lambda_reg / (2 * m)) * np.sum(theta[1:] ** 2)  # Regularization (exclude bias term)
    return mse_loss + reg_loss

#=============================================
# Define Gradient Computing
#=============================================
def gradient_computing(X_bias, y, theta, lambda_reg):
    m = len(y)
    #linear_part = X_bias @ theta
    #predictions = np.log(1 + np.exp(linear_part)) 
    #sigma = 1 / (1 + np.exp(-linear_part))  # sigmoid
    predictions = X_bias @ theta
    errors = predictions - y
    #gradients = (1 / m) * (X_bias.T @ (errors * sigma))  # modify gradient
    gradients = (1 / m) * (X_bias.T @ errors)
    gradients[1:] += (lambda_reg / m) * theta[1:]
    return gradients
 
#=============================================
# Define Gradient Descent Function
#=============================================
def gradient_descent(X_bias, y, learning_rate, iteration, lambda_reg, tolerance):
    m, n = X_bias.shape # m = number of samples, n = number of features
    theta = np.zeros(n) # initialize theta
    loss_history = [] # track loss over iteration
    theta_history = []
    track_iterations = [0, 10, 100, 500, 1000]
    
    for i in range(iteration):
        Loss = loss(X_bias, y, theta, lambda_reg)
        gradient = gradient_computing(X_bias, y, theta, lambda_reg)
        theta -= learning_rate * gradient
        loss_history.append(Loss)
        if i > 0 and abs(Loss - loss_history[i - 1]) < tolerance: # Model considered optimal when the change in loss is very small
            print(f"Converged at iteration {i}")
            break
        
        if i in track_iterations:
            print(f"Iteration {i}: Loss = {Loss}, Theta = {theta}")
            theta_history.append((i, Loss, theta.copy()))
    
    return theta, loss_history, theta_history

#=============================================
# Define K-fold Cross-validation
#=============================================
def k_fold_cross_validation(X, y, k, learning_rate, iteration, lambda_reg, tolerance):
    kf = KFold(n_splits = k, shuffle = True, random_state = 16)
    mse_list = [] 
    theta_list = []
    all_loss_histories = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Add bias term
        X_train_bias = np.column_stack([np.ones(X_train.shape[0]), X_train[:, 0], np.sin(X_train[:, 1]), np.cos(X_train[:, 1]), X_train[:, 2:]])
        X_test_bias = np.column_stack([np.ones(X_test.shape[0]), X_test[:, 0], np.sin(X_test[:, 1]), np.cos(X_test[:, 1]), X_test[:, 2:]])
        # Train the model
        theta, loss_history, theta_history = gradient_descent(X_train_bias, y_train, learning_rate, iteration, lambda_reg, tolerance)
        theta_list.append(theta)
        all_loss_histories.append(loss_history)
        # Make predictions
        y_prediction = X_test_bias @ theta
        # Compute Mean Squared Error for the fold
        mse = mean_squared_error(y_test, y_prediction)
        mse_list.append(mse)
    
    # Find the best theta based on minimum MSE
    best_fold_index = np.argmin(mse_list)
    best_theta = theta_list[best_fold_index]
    
    # Plot combined loss histories for all folds
    average_loss_history = np.mean(all_loss_histories, axis=0)
    plt.plot(average_loss_history, label="Average Loss Across Folds")
    plt.legend()
    plt.show()

    print(f"Best fold index: {best_fold_index + 1}, MSE: {mse_list[best_fold_index]}")
    return mse_list, np.mean(mse_list), best_theta # Return average mse and best theta

#=============================================
# Initialize Parameters
#=============================================
k = 5
iteration = 1000
learning_rate = 0.1
lambda_reg = 0.1
tolerance = 1e-10

#=============================================
# Output results
#=============================================
# Perform K-Fold Cross Validation
mse_list, average_mse, best_theta = k_fold_cross_validation(
    X_normalized, Y, k, learning_rate, iteration, lambda_reg, tolerance
)
print("MSE for each fold:", mse_list)
print("Average MSE across folds:", average_mse)
print("Best theta based on validation performance:", best_theta)
corr = data.corr()
print(corr["Extent (km2)"])

# Load 2022-2023 data
new_data = pd.read_excel("Monthly_Predicted_Data_for_2025.xlsx")

# Preprocess new data
new_data["Month"] = new_data["Month"].map(month_to_num)
new_data['Month_angle'] = (new_data['Month'] - 1) * 2 * np.pi / 12
new_data['Solar Radiation(J/m2)'] = new_data['Solar Radiation(J/m2)'].apply(lambda x: np.log10(x + 1e-12)) # Logarise solar radiation

X_new = new_data[['Year', 'Month_angle', 'Temperature(C)', 'Solar Radiation(J/m2)', 'Average Salinity(PSU)']].values

# Normalize the new data
X_new_normalized = (X_new - X_mean) / X_std

# Add bias term and transform features
X_new_bias = np.column_stack([
    np.ones(X_new_normalized.shape[0]), 
    X_new_normalized[:, 0], 
    np.sin(X_new_normalized[:, 1]), 
    np.cos(X_new_normalized[:, 1]), 
    X_new_normalized[:, 2:]
])

# Generate predictions
y_pred_new = X_new_bias @ best_theta

# Add predictions to the new data for visualization
new_data["Predicted Extent (km2)"] = y_pred_new

# Plot predictions and actual values (if available)
plt.figure(figsize=(10, 6))

# Assuming you have actual values for validation
if "Extent (km2)" in new_data.columns:
    plt.plot(new_data["Year"] + new_data["Month"] / 12, new_data["Extent (km2)"], label="Actual Extent", marker="o")

plt.plot(new_data["Year"] + new_data["Month"] / 12, new_data["Predicted Extent (km2)"], label="Predicted Extent", linestyle="--", marker="x")

plt.xlabel("Year")
plt.ylabel("Sea Ice Extent (km2)")
plt.title("Predicted Sea Ice Extent(2025)")
plt.legend()
plt.grid(True)
plt.show()

new_data.to_excel("sea_ice_predictions_2025.xlsx", index=False)
print("Predictions saved to sea_ice_predictions_2025.xlsx")

