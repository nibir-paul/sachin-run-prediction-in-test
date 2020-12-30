import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

matches = pd.read_csv("input/Test.csv")
print("\nHaving a look at the dataset - ")
print(matches.head())
print("\nChecking the shape of the dataset - ")
print(matches.shape)

# Since few columns have '-' as missing values its replaced with Nan for better understanding
# Also since the numbers are string its converted to float type
columns = ['Mins', 'BF', '4s', '6s', 'SR', 'Pos', 'Inns']
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

# Encoding of Opposition and Ground for better reading
encoding_features = ['Opposition','Ground']
labelencoder = LabelEncoder()
for feature in encoding_features:
    matches[feature] = labelencoder.fit_transform(matches[feature])
    matches[feature] = matches[feature].astype('float')

# Features selected
matches = matches[['Runs', 'Inns', 'Opposition', 'Ground']]

# Dropping rows with Nan values as they are not needed
matches.dropna(inplace=True)
print("\nFinal check so that there are no missing values - ")
print(matches.apply(lambda x: sum(x.isnull()),axis=0))
print("\nData for model building - ")
print(matches.head())

# Applying scaler since some columns have lower integer values
scaler = MinMaxScaler()
features = ['Runs', 'Inns', 'Opposition', 'Ground']
matches[features] = scaler.fit_transform(matches[features])

# Selection of features for train test split
features = matches[['Inns', 'Opposition', 'Ground']]
target = matches[['Runs']]

# We specify random seed so that the train and test data set always have the same rows, respectively
np.random.seed(0)
x_train, x_test,y_train,y_test = train_test_split(features, target, train_size = 0.7, test_size = 0.3, random_state = 100)

# Building of multiple linear regression model
linearregression = LinearRegression()
model = linearregression.fit(x_train, y_train)
predictions = linearregression.predict(x_test)
print("\nAccuracy of the model - ")
print(linearregression.score(x_test, y_test)*1000)