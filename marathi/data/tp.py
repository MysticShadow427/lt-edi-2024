import pandas as pd

df = pd.read_csv('/Users/devendrakayande/Desktop/lt-edi/telugu/data/homo_Telugu_test.csv')
print(df['Category'].value_counts())