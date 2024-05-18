import pandas as pd

train_df = pd.read_csv('/content/lt-edi-2024/marathi/data/Marathi_train.csv')
val_df = pd.read_csv('/content/lt-edi-2024/marathi/data/Marathi_dev.csv')
test_df = pd.read_csv('/content/lt-edi-2024/marathi/data/Marathi_test.csv')

train_df = pd.concat([train_df, val_df])
train_df.rename(columns={'Text': 'content','Category ':'label'}, inplace=True)
test_df.rename(columns={'Text': 'content','Category ':'label'}, inplace=True)
# Rename the columns to match the desired JSON format
import numpy as np

def random_undersample(df,random_seed = 42,desired_samples = 500):
    np.random.seed(random_seed)
    class_1_df = df[df['label'] == 'None of the categories']
    class_2_df = df[df['label'] == 'Homophobia']
    class_3_df = df[df['label'] == 'Transphobia']

    if len(class_1_df) > desired_samples:
        undersampled_class_1_df = class_1_df.sample(n=desired_samples, replace=True)
    else:
        undersampled_class_1_df = class_1_df.sample(frac=1.0)   

    augmented_df = pd.concat([undersampled_class_1_df, class_2_df,class_3_df])

    augmented_df = augmented_df.sample(frac=1.0).reset_index(drop=True)

    return augmented_df

train_df = random_undersample(train_df)

train_df.rename(columns={'content': 'text', 'label': 'label'}, inplace=True)
test_df.rename(columns={'content': 'text', 'label': 'label'}, inplace=True)
# Convert the DataFrame to a JSON string
json_str_train = train_df.to_json(orient='records')
json_str_test = test_df.to_json(orient='records')
# Write the JSON string to a file
with open('/content/Dual-Contrastive-Learning/data/MAR_Train.json', 'w') as f:
    f.write(json_str_train)
with open('/content/Dual-Contrastive-Learning/data/MAR_Test.json', 'w') as f:
    f.write(json_str_test)