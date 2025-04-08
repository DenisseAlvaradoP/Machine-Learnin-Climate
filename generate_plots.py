# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Function to load and clean the data
def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=',')  # Load the dataset from a CSV file
    df.columns = df.columns.str.strip()  # Remove any leading or trailing spaces from column names
    df.replace(-999, pd.NA, inplace=True)  # Replace placeholder values (-999) with NaN (missing value indicator)
    df.dropna(inplace=True)  # Drop rows with missing values
    df['MESS_DATUM'] = pd.to_datetime(df['MESS_DATUM'], format='%Y%m%d')  # Convert date column to datetime format
    # Rename columns for better readability and understanding
    df.columns = [
        'STATIONS_ID', 'Date', 'QN_3', 'FX', 'Wind Speed (m/s)', 'QN_4', 'Precipitation Level (mm)', 'RSKF', 
        'Sun Duration (hours)', 'Snow Height (cm)', 'Cloud Cover (octaves)', 'Vapor Pressure (hPa)', 
        'Atmospheric Pressure (hPa)', 'Average Temperature (°C)', 'Relative Humidity (%)', 'Maximum Temperature (°C)', 
        'Minimum Temperature (°C)', 'Soil Minimum Temperature (°C)', 'eor'
    ]
    df = df[df['RSKF'] != 9]  # Filter out rows where the 'RSKF' value is 9 (likely an outlier or unwanted value)
    df['Month'] = df['Date'].dt.month  # Extract the month from the 'Date' column and create a 'Month' column
    return df  # Return the cleaned and processed DataFrame

# Function to apply KMeans clustering to selected variables
def apply_clustering(df, variables, n_clusters=3):
    data_selected = df[variables].copy()  # Select the relevant variables for clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # Initialize KMeans with the specified number of clusters
    data_selected['Cluster'] = kmeans.fit_predict(data_selected)  # Fit KMeans and assign cluster labels to the data
    # Combine the cluster data with the original date and month for further analysis
    data_final = pd.concat([data_selected, df[['Date', 'Month']]], axis=1)
    return data_final, kmeans  # Return the clustered data and the KMeans model

# Function to calculate the range (min and max) for each cluster for a specified variable
def get_cluster_ranges(data_final, kmeans, variable):
    cluster_ranges = {}
    # Iterate over each cluster to calculate the min and max values for the selected variable
    for cluster in range(kmeans.n_clusters):
        cluster_data = data_final[data_final['Cluster'] == cluster]  # Get data for the current cluster
        min_val = cluster_data[variable].min()  # Calculate the minimum value for the variable in this cluster
        max_val = cluster_data[variable].max()  # Calculate the maximum value for the variable in this cluster
        cluster_ranges[cluster] = {'min': min_val, 'max': max_val}  # Store the min and max values in a dictionary
        # Add columns to the data for the min and max values of the variable within each cluster
        data_final.loc[data_final['Cluster'] == cluster, f'{variable} Min'] = min_val
        data_final.loc[data_final['Cluster'] == cluster, f'{variable} Max'] = max_val
    return data_final, cluster_ranges  # Return the updated data and the cluster ranges

# Function to label clusters based on their ranges
def label_clusters(cluster_ranges, labels):
    # Sort clusters by their minimum value to map them to meaningful labels
    sorted_clusters = sorted(cluster_ranges.items(), key=lambda x: x[1]['min'])
    # Map sorted clusters to the provided labels
    cluster_labels = {cluster: label for cluster, label in zip([c[0] for c in sorted_clusters], labels)}
    return cluster_labels  # Return the dictionary mapping clusters to labels

# Function to generate a scatter plot of clusters
def generate_scatter_plot(data_final, x_var, y_var, cluster_labels, title, month_labels=False):
    colors = {0: 'blue', 1: 'green', 2: 'red'}  # Define colors for each cluster

    plt.figure(figsize=(12, 6))  # Set the size of the plot
    for cluster, color in colors.items():
        subset = data_final[data_final['Cluster'] == cluster]  # Get data for the current cluster
        # Plot a scatter plot for each cluster
        plt.scatter(subset[x_var], subset[y_var], c=color, label=cluster_labels[cluster], alpha=0.6)

    plt.title(title)  # Set the title of the plot
    plt.xlabel(x_var)  # Set the label for the x-axis
    plt.ylabel(y_var)  # Set the label for the y-axis
    plt.legend()  # Show the legend
    
    if month_labels:  # If month labels are requested, set custom labels for the x-axis
        plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    plt.show()  # Display the plot

# Main function to execute the clustering and plotting
def main():
    file_path = 'fulldata.csv' # Path to the input data file
    df = load_data(file_path)  # Load and clean the data

    # Apply clustering and generate a plot for temperature-related variables
    data_final_1, kmeans_1 = apply_clustering(df, ['Average Temperature (°C)', 'Maximum Temperature (°C)', 'Minimum Temperature (°C)'], n_clusters=3)
    data_final_1, cluster_ranges_1 = get_cluster_ranges(data_final_1, kmeans_1, 'Average Temperature (°C)')
    cluster_labels_1 = label_clusters(cluster_ranges_1, ['Cold temperature', 'Cool temperature', 'Hot temperature'])
    data_final_1['Cluster Label'] = data_final_1['Cluster'].map(cluster_labels_1)  # Map cluster labels to the data
    # Merge with the original data to include cluster information
    all_data = df.merge(data_final_1[['Date', 'Cluster', 'Cluster Label', 'Average Temperature (°C) Min', 'Average Temperature (°C) Max']], on='Date', suffixes=('', '_Temp'))
    # Generate a scatter plot for temperature clusters by month
    generate_scatter_plot(data_final_1, 'Month', 'Average Temperature (°C)', cluster_labels_1, 'Temperature Clusters by Month and Average Temperature', month_labels=True)

    # Apply clustering and generate a plot for sun duration and cloud cover
    data_final_2, kmeans_2 = apply_clustering(df, ['Sun Duration (hours)', 'Cloud Cover (octaves)'], n_clusters=3)
    data_final_2, cluster_ranges_2 = get_cluster_ranges(data_final_2, kmeans_2, 'Sun Duration (hours)')
    cluster_labels_2 = label_clusters(cluster_ranges_2, ['Low sun duration', 'Medium sun duration', 'High sun duration'])
    data_final_2['Cluster Label'] = data_final_2['Cluster'].map(cluster_labels_2)  # Map cluster labels to the data
    # Merge with the original data to include cluster information
    all_data = all_data.merge(data_final_2[['Date', 'Cluster', 'Cluster Label', 'Sun Duration (hours) Min', 'Sun Duration (hours) Max']], on='Date', suffixes=('', '_Sun'))
    # Generate a scatter plot for sun duration clusters by cloud cover
    generate_scatter_plot(data_final_2, 'Sun Duration (hours)', 'Cloud Cover (octaves)', cluster_labels_2, 'Sun Duration Clusters by Cloud Cover')

    # Apply clustering and generate a plot for precipitation level and cloud cover
    data_final_3, kmeans_3 = apply_clustering(df, ['Precipitation Level (mm)', 'Cloud Cover (octaves)'], n_clusters=3)
    data_final_3, cluster_ranges_3 = get_cluster_ranges(data_final_3, kmeans_3, 'Precipitation Level (mm)')
    cluster_labels_3 = label_clusters(cluster_ranges_3, ['Low precipitation', 'Medium precipitation', 'High precipitation'])
    data_final_3['Cluster Label'] = data_final_3['Cluster'].map(cluster_labels_3)  # Map cluster labels to the data
    # Merge with the original data to include cluster information
    all_data = all_data.merge(data_final_3[['Date', 'Cluster', 'Cluster Label', 'Precipitation Level (mm) Min', 'Precipitation Level (mm) Max']], on='Date', suffixes=('', '_Precip'))
    # Generate a scatter plot for precipitation clusters by cloud cover
    generate_scatter_plot(data_final_3, 'Precipitation Level (mm)', 'Cloud Cover (octaves)', cluster_labels_3, 'Precipitation Clusters by Cloud Cover')

    # Apply clustering and generate a plot for cloud cover and precipitation level
    data_final_5, kmeans_5 = apply_clustering(df, ['Cloud Cover (octaves)', 'Precipitation Level (mm)'], n_clusters=3)
    data_final_5, cluster_ranges_5 = get_cluster_ranges(data_final_5, kmeans_5, 'Cloud Cover (octaves)')
    cluster_labels_5 = label_clusters(cluster_ranges_5, ['Low Cloud Cover', 'Medium Cloud Cover', 'High Cloud Cover'])
    data_final_5['Cluster Label'] = data_final_5['Cluster'].map(cluster_labels_5)  # Map cluster labels to the data
    # Merge with the original data to include cluster information
    all_data = all_data.merge(data_final_5[['Date', 'Cluster', 'Cluster Label', 'Cloud Cover (octaves) Min', 'Cloud Cover (octaves) Max']], on='Date', suffixes=('', '_Clouds'))
    # Generate a scatter plot for cloud cover clusters by precipitation level
    generate_scatter_plot(data_final_5, 'Cloud Cover (octaves)', 'Precipitation Level (mm)', cluster_labels_5, 'Cloud Cover by Precipitation Level')

    # Apply clustering and generate a plot for snow height and average temperature
    data_final_6, kmeans_6 = apply_clustering(df, ['Snow Height (cm)', 'Average Temperature (°C)'], n_clusters=3)
    data_final_6, cluster_ranges_6 = get_cluster_ranges(data_final_6, kmeans_6, 'Snow Height (cm)')
    cluster_labels_6 = label_clusters(cluster_ranges_6, ['High snow', 'Light snow', 'No snow'])
    data_final_6['Cluster Label'] = data_final_6['Cluster'].map(cluster_labels_6)  # Map cluster labels to the data
    # Merge with the original data to include cluster information
    all_data = all_data.merge(data_final_6[['Date', 'Cluster', 'Cluster Label', 'Snow Height (cm) Min', 'Snow Height (cm) Max']], on='Date', suffixes=('', '_Snow_Temp'))
    # Generate a scatter plot for snow clusters by average temperature
    generate_scatter_plot(data_final_6, 'Average Temperature (°C)', 'Snow Height (cm)', cluster_labels_6, 'Snow Clusters by Average Temperature')

    # Apply clustering and generate a plot for atmospheric pressure and wind speed
    data_final_7, kmeans_7 = apply_clustering(df, ['Atmospheric Pressure (hPa)', 'Wind Speed (m/s)'], n_clusters=3)
    data_final_7, cluster_ranges_7 = get_cluster_ranges(data_final_7, kmeans_7, 'Atmospheric Pressure (hPa)')
    cluster_labels_7 = label_clusters(cluster_ranges_7, ['Low Pressure', 'Medium Pressure', 'High Pressure'])
    data_final_7['Cluster Label'] = data_final_7['Cluster'].map(cluster_labels_7)  # Map cluster labels to the data
    # Merge with the original data to include cluster information
    all_data = all_data.merge(data_final_7[['Date', 'Cluster', 'Cluster Label', 'Atmospheric Pressure (hPa) Min', 'Atmospheric Pressure (hPa) Max']], on='Date', suffixes=('', '_Pressure_Wind'))
    # Generate a scatter plot for atmospheric pressure clusters by wind speed
    generate_scatter_plot(data_final_7, 'Atmospheric Pressure (hPa)', 'Wind Speed (m/s)', cluster_labels_7, 'Pressure by Wind')

    # Apply clustering and generate a plot for wind speed and atmospheric pressure
    data_final_8, kmeans_8 = apply_clustering(df, ['Wind Speed (m/s)', 'Atmospheric Pressure (hPa)'], n_clusters=3)
    data_final_8, cluster_ranges_8 = get_cluster_ranges(data_final_8, kmeans_8, 'Wind Speed (m/s)')
    cluster_labels_8 = label_clusters(cluster_ranges_8, ['Low Wind', 'Medium Wind', 'High Wind'])
    data_final_8['Cluster Label'] = data_final_8['Cluster'].map(cluster_labels_8)  # Map cluster labels to the data
    # Merge with the original data to include cluster information
    all_data = all_data.merge(data_final_8[['Date', 'Cluster', 'Cluster Label', 'Wind Speed (m/s) Min', 'Wind Speed (m/s) Max']], on='Date', suffixes=('', '_Wind_Pressure'))
    # Generate a scatter plot for wind speed clusters by atmospheric pressure
    generate_scatter_plot(data_final_8, 'Wind Speed (m/s)', 'Atmospheric Pressure (hPa)', cluster_labels_8, 'Wind by Pressure')

    # Save the final combined data with all cluster labels to a new CSV file
    all_data.to_csv('final_data_with_clusters.csv', index=False)

# Check if the script is being run directly
if __name__ == "__main__":  # This condition is used to prevent code from running when the module is imported
    main()  # Call the main function to execute the script
