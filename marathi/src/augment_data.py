import numpy as np
import pandas as pd

def random_undersample(df,random_seed = 42,desired_samples = 450):
    np.random.seed(random_seed)
    class_1_df = df[df['Category '] == 'None of the categories']
    class_2_df = df[df['Category '] == 'Homophobia']
    class_3_df = df[df['Category '] == 'Transphobia']

    if len(class_1_df) > desired_samples:
        undersampled_class_1_df = class_1_df.sample(n=desired_samples, replace=True)
    else:
        undersampled_class_1_df = class_1_df.sample(frac=1.0)   

    augmented_df = pd.concat([undersampled_class_1_df, class_2_df,class_3_df])

    augmented_df = augmented_df.sample(frac=1.0).reset_index(drop=True)

    return augmented_df