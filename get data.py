"""
Parameters:
STATIONS_ID: Station ID.
MESS_DATUM: Measurement date in YYYYMMDD format.
QN_3: Data quality (numeric code).
FX: Maximum wind speed (m/s).
FM: Average wind speed (m/s).
QN_4: Data quality (numeric code).
RSK: Daily precipitation height (mm).
RSKF: Precipitation form (numeric code).
SDK: Daily sunshine duration (hours).
SHK_TAG: Daily snow height (cm).
NM: Daily cloud cover (in eighths).
VPM: Daily vapor pressure (hPa).
PM: Daily atmospheric pressure (hPa).
TMK: Daily average temperature (°C).
UPM: Daily relative humidity (%).
TXK: Daily maximum temperature (°C).
TNK: Daily minimum temperature (°C).
TGK: Minimum ground level temperature (°C).
eor: End of record.
"""
import os
import pandas as pd

# Path to the text file
txt_file_path = r"C:\Users\denis\Documents\Documents\PROGRAMACIÓN\Machine Learning\"

# Read the text file
df = pd.read_csv(txt_file_path, delimiter=';')

# Full path of the CSV file, including the file name
csv_file_path = r'C:\Users\denis\Documents\Documents\PROGRAMACIÓN\Machine Learning\fulldata.csv'

# Check if the directory exists, and if not, create it
directory = os.path.dirname(csv_file_path)
if not os.path.exists(directory):
    os.makedirs(directory)

# Save the DataFrame as a CSV file
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved successfully at {csv_file_path}")