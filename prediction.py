import pandas as pd
import numpy as np

matches = pd.read_csv("input/Test.csv")
print("\nHaving a look at the dataset - ")
print(matches.head())
print("\nChecking the shape of the dataset - ")
print(matches.shape)

# Since few columns have '-' as missing values its replaced with Nan for better understanding
# Also since the numbers are string its converted to float type
columns = ['Mins', 'BF', '4s', '6s', 'SR', 'Pos']
for column in columns:
    matches[column][matches[column] == '-'] = np.nan
    matches[column] = matches[column].astype('float')
print("\nDescription of the dataset - ")
print(matches.describe())

# DNB and TDNB implies that the batsman did not bat so its replaced with Nan
matches.loc[matches.Runs == "DNB", "Runs"] = np.nan
matches.loc[matches.Runs == "TDNB", "Runs"] = np.nan
matches['Runs'] = matches['Runs'].astype('float')
print("\nChecking if DNB and TDNB were replaced by null values - ")
print(matches.tail(6))

# The v infront of every opposition team is replaced for better readability
for row in range(matches.shape[0]):
    matches.Opposition[row] = matches.Opposition[row].replace('v ','')
print("\nFinal cleaned data - ")
print(matches.head())