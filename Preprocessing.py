
# Pandas
import pandas as pd

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

# Utility
from utility import convert_to_seconds

# Torch
import torch
from torch.utils.data import DataLoader, random_split, Subset

class Prepropressing:    
    def __init__(self):
        pass
    
    def first_exemple():
        
        dataset = pd.read_csv("./202207-divvy-tripdata.csv") # Replace by your dataset
        data = dataset.copy() # Copy

        data_prep = data.copy()

        # Convert the Series "ride_length" to second
        data_prep["ride_length"] = data_prep["ride_length"].apply(convert_to_seconds)

        # dropNaN
        data_prep = data_prep.dropna()
        
        data_prep['ride_length'] = data_prep['ride_length'].astype('int64')
        
        features = ['start_lat', 
                'start_lng',
                'end_lat',
                'end_lng',
                'member_casual',
                'ride_length',
                'day_of_week',
                'ride_id']
                
        target = ["rideable_type"]

        data_prep = data_prep.loc[(data_prep["rideable_type"] == "classic_bike") | (data_prep["rideable_type"] == "electric_bike")]
        X = data_prep[features]
        y = data_prep[target]

        y_OH = OneHotEncoder(handle_unknown='ignore', sparse=False).fit_transform(y)

        num_cols = ["start_lat", "start_lng", "end_lat", "end_lng", "day_of_week", "ride_length"]
        categorical_cols_less_values = ["member_casual"]


        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat_less_values', OrdinalEncoder(), categorical_cols_less_values)
            ])

        X_prep = preprocessor.fit_transform(X)
        
        X_tensor = torch.tensor(X_prep, dtype=torch.float32)
        y_tensor = torch.tensor(y_OH, dtype=torch.float32)
        
        return torch.utils.data.TensorDataset(X_tensor, y_tensor)