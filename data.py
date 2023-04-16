import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


'''
 data_loader: Used to Prepare Data For MLP in a dataframe format
 
    Inputs:
        file (str): CSV file of dataset provided by Building Performance Dataset
        encoder (Encoder): used to encode all categorical variables
        label (str): Type of EUI used as label
    
    Output:
        df (DataFrame): data formatted and cleaned in DataFrame
        labels (DataFrame): labels of data

'''
def data_loader(file, encoder):

    # Import Dataset
    df = pd.read_csv(file)

    # Separate Labels and Categorical Features
    labels = df["site_eui"]
    categorical_feats = ['State_Factor','building_class', 'facility_type']

    #Encode all Categorical Features
    df[categorical_feats] = df[categorical_feats].apply(encoder.fit_transform)

    #Replace all "Unknown" and "No Value" into -1
    df = df.fillna(-1)
    labels = labels.fillna(-1)

    return df, labels


'''
 data_visualization: Used to Visualize Data

    Inputs:
        file (str): CSV file of dataset provided by Building Performance Dataset

    Output:
        None

'''
def data_visualization(file):

    df = pd.read_csv(file, na_values=['Unknown', 'No Value'])
    print(f'floor_area: {df["floor_area"]}')
    print(f'site_eui: {df["site_eui"]}')

    df.plot.scatter(x="site_eui", y="floor_area")
    plt.show()





#Initialize Variables for Data Loading
file = "Data/train.csv"
encoder = LabelEncoder()

df, labels = data_loader(file, encoder)

#Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2)

#print(f'X_train: {X_train}')
#print(f'y_train: {y_train}')

#Train simple MLP (Multi layer perceptron) Regression
model = MLPRegressor(max_iter=500).fit(X_train, y_train)
y_pred = model.predict(X_test)

y_pred = np.array(y_pred)
y_test = np.array(y_test)

print(f'y_pred: {y_pred}')
print(f'y_test: {y_test}')

#Evaluate Results
print(f'R^2 Score: {r2_score(y_test, y_pred)}')





