import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to load and clean the data
def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=',')  # Load the dataset from a CSV file
    df.columns = df.columns.str.strip()  # Remove any leading or trailing spaces from the column names
    df.replace(-999, pd.NA, inplace=True)  # Replace placeholder values (-999) with NaN (missing value indicator)
    df.dropna(inplace=True)  # Drop rows with missing values to clean the data

    # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])  # Convert the 'Date' column to datetime format
    df['Month'] = df['Date'].dt.month  # Extract the month from the date and create a new column 'Month'

    # Encoding cluster information as numerical values
    cluster_mapping = {'Low': 1, 'Medium': 2, 'High': 3}  # Define a mapping for categorical cluster labels to numerical values
    df['Cluster'] = df['Cluster Label'].map(cluster_mapping)  # Map the cluster labels to numerical values
    df['Cluster_Sun'] = df['Cluster Label_Sun'].map(cluster_mapping)
    df['Cluster_Precip'] = df['Cluster Label_Precip'].map(cluster_mapping)
    df['Cluster_Clouds'] = df['Cluster Label_Clouds'].map(cluster_mapping)
    df['Cluster_Snow_Temp'] = df['Cluster Label_Snow_Temp'].map(cluster_mapping)
    df['Cluster_Pressure_Wind'] = df['Cluster Label_Pressure_Wind'].map(cluster_mapping)
    df['Cluster_Wind_Pressure'] = df['Cluster Label_Wind_Pressure'].map(cluster_mapping)

    return df  # Return the cleaned and processed DataFrame

# Function to train and save the model
def train_and_save_model(df, features, target, model_path):
    X = df[features]  # Select the features (independent variables)
    y = df[target]  # Select the target variable (dependent variable)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()  # Initialize a linear regression model
    model.fit(X_train, y_train)  # Train the model on the training data

    # Save the trained model to a file
    joblib.dump(model, model_path)
    print(f'Model saved to {model_path}')

    # Plotting the results
    y_pred = model.predict(X_test)  # Predict the target variable for the test set

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)  # Scatter plot of actual vs predicted values
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3)  # Plot a diagonal line (y=x) for reference
    plt.xlabel('Measured')  # X-axis label
    plt.ylabel('Predicted')  # Y-axis label
    plt.title('Measured vs Predicted Values')  # Plot title
    plt.show()  # Display the plot

    # Return evaluation metrics
    mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)  # Calculate Mean Absolute Error
    r2 = r2_score(y_test, y_pred)  # Calculate R² score (coefficient of determination)

    return mse, mae, r2  # Return the calculated metrics

# Main function to execute the script
def main():
    file_path = 'final_data_with_clusters.csv'
    model_path = 'linear_regression_model_with_clusters.pkl'
    
    df = load_data(file_path)
    print("Data loaded successfully.")  # depuration message

    features = ['Month', 'Wind Speed (m/s)', 'Precipitation Level (mm)', 'Sun Duration (hours)', 
                'Snow Height (cm)', 'Cloud Cover (octaves)', 'Vapor Pressure (hPa)', 
                'Atmospheric Pressure (hPa)', 'Relative Humidity (%)']
    target = 'Average Temperature (°C)'
    
    mse, mae, r2 = train_and_save_model(df, features, target, model_path)
    
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R² Score: {r2}')

if __name__ == "__main__":
    main()