import pandas as pd

train_df = pd.read_csv('/content/lt-edi-2024/telugu/data/homo_Telugu_train.csv')
val_df = pd.read_csv('/content/lt-edi-2024/telugu/data/homo_Telugu_dev.csv')
test_df = pd.read_csv('/content/lt-edi-2024/telugu/data/homo_Telugu_test.csv')

# train_df = pd.concat([train_df, val_df])
train_df.rename(columns={'Text': 'content','Category ':'label'}, inplace=True)
test_df.rename(columns={'Text': 'content','Category ':'label'}, inplace=True)
# Rename the columns to match the desired JSON format
import numpy as np


train_df.rename(columns={'content': 'text', 'label': 'label'}, inplace=True)
test_df.rename(columns={'content': 'text', 'label': 'label'}, inplace=True)
# Convert the DataFrame to a JSON string
json_str_train = train_df.to_json(orient='records')
json_str_test = test_df.to_json(orient='records')
# Write the JSON string to a file
with open('/content/Dual-Contrastive-Learning/data/TEL_Train.json', 'w') as f:
    f.write(json_str_train)
with open('/content/Dual-Contrastive-Learning/data/TEL_Test.json', 'w') as f:
    f.write(json_str_test)