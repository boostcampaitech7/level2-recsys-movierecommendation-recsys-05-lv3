import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop('time', axis=1)
    
    user_id_map = {id: i for i, id in enumerate(df['user'].unique())}
    item_id_map = {id: i for i, id in enumerate(df['item'].unique())}
    
    df['user'] = df['user'].map(user_id_map)
    df['item'] = df['item'].map(item_id_map)
    
    train_data, val_data = train_test_split(df, test_size=0.2, stratify=df['user'], random_state=42)
    
    return train_data, val_data, user_id_map, item_id_map

base_path = '/data/ephemeral/home/ryu/data/'
input_file = os.path.join(base_path, 'train', 'train_ratings.csv')
output_path = os.path.join(base_path, 'processed')

os.makedirs(output_path, exist_ok=True)

train_data, val_data, user_id_map, item_id_map = preprocess_data(input_file)

train_data.to_csv(os.path.join(output_path, 'processed_train_data.csv'), index=False)
val_data.to_csv(os.path.join(output_path, 'processed_val_data.csv'), index=False)

pd.DataFrame(list(user_id_map.items()), columns=['original_id', 'mapped_id']).to_csv(os.path.join(output_path, 'user_id_map.csv'), index=False)
pd.DataFrame(list(item_id_map.items()), columns=['original_id', 'mapped_id']).to_csv(os.path.join(output_path, 'item_id_map.csv'), index=False)