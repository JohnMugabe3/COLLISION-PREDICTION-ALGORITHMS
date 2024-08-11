import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


base_path = '/content/drive/MyDrive/UAV_DELIVERY_DATASET_FULL'
folders = ['UAV_DELIVERY_DATASET_1', 'UAV_DELIVERY_DATASET_2', 'UAV_DELIVERY_DATASET_3']


def read_log_files(base_path, folders):
    headers = ["simt", "id", "type", "lat", "lon", "alt", "tas", "cas", "vs", "gs", "distflown",
               "Temp", "trk", "hdg", "p", "rho", "thrust", "drag", "phase", "fuelflow"]
    data = []
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.log'):
                file_path = os.path.join(folder_path, file_name)
                try:
                   
                    df = pd.read_csv(file_path, delimiter=',', names=headers, skiprows=1, comment='#')
                    data.append(df)
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")
    if data:
        return pd.concat(data, ignore_index=True)
    else:
        return None


data = read_log_files(base_path, folders)
if data is not None:
    print("Data columns:", data.columns)
else:
    print("No data loaded.")


def extract_relevant_columns(df):
    if df is not None:
        return df[['lat', 'lon', 'alt']].dropna()
    else:
        return None

data = extract_relevant_columns(data)


def normalize_data(df):
    if df is not None:
        try:
           
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
            df['alt'] = pd.to_numeric(df['alt'], errors='coerce')

          
            df = df.dropna(subset=['lat', 'lon', 'alt']).copy()

           
            scaler = MinMaxScaler()
            df[['lat', 'lon', 'alt']] = scaler.fit_transform(df[['lat', 'lon', 'alt']])
            return df
        except Exception as e:
            print("Failed to normalize data:", e)
            return None
    else:
        return None

if data is not None:
    print("Data types before normalization:", data.dtypes)
    print("Sample data before normalization:", data[['lat', 'lon', 'alt']].head())
    data = normalize_data(data)

if data is not None:
    print(data.head())
else:
    print("No data to display.")
