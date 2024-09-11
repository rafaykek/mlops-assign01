import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def saveModel():
    file_path = '/kaggle/input/housing-prices-dataset/Housing.csv'
    df = pd.read_csv(file_path)
    
    categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, 'linear_regression_model.pkl')
    print("Model saved as 'linear_regression_model.pkl'")

def predictPrice(area, bedrooms, bathrooms, stories, parking, mainroad_yes, guestroom_yes, basement_yes, hotwaterheating_yes, airconditioning_yes, prefarea_yes, furnishingstatus_semi_furnished, furnishingstatus_unfurnished):

    model = joblib.load('linear_regression_model.pkl')
    
    input_features = np.array([[area, bedrooms, bathrooms, stories, parking, mainroad_yes, guestroom_yes, basement_yes, hotwaterheating_yes, airconditioning_yes, prefarea_yes, furnishingstatus_semi_furnished, furnishingstatus_unfurnished]])
    
    predicted_price = model.predict(input_features)
    
    return predicted_price