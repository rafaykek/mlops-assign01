import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def saveModel():
    file_path = 'mlops-assign01/Housing.csv'
    df = pd.read_csv(file_path)
    
    categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    try:
        joblib.dump(model, 'linear_regression_model.pkl')
        print("Model saved successfully as 'linear_regression_model.pkl'")
    except Exception as e:
        print(f"Error while saving the model: {e}")

def predictPrice(area, bedrooms, bathrooms, stories, parking, mainroad_yes, guestroom_yes, basement_yes, hotwaterheating_yes, airconditioning_yes, prefarea_yes, furnishingstatus_semi_furnished, furnishingstatus_unfurnished):

    if area <= 0 or bedrooms < 0 or bathrooms < 0 or stories < 0 or parking < 0:
        raise ValueError("Invalid input values provided.")

    model = joblib.load('linear_regression_model.pkl')

    feature_names = model.feature_names_in_

    input_data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,
        'mainroad_yes': mainroad_yes,
        'guestroom_yes': guestroom_yes,
        'basement_yes': basement_yes,
        'hotwaterheating_yes': hotwaterheating_yes,
        'airconditioning_yes': airconditioning_yes,
        'prefarea_yes': prefarea_yes,
        'furnishingstatus_semi-furnished': furnishingstatus_semi_furnished, 
        'furnishingstatus_unfurnished': furnishingstatus_unfurnished,
    }

    input_features = pd.DataFrame([input_data], columns=feature_names)
    
    predicted_price = model.predict(input_features)
    
    return predicted_price

if __name__ == '__main__':
    saveModel()
