from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)


model = joblib.load('linear_regression_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    stories = int(request.form['stories'])
    parking = int(request.form['parking'])
  
    mainroad_yes = 1 if 'mainroad_yes' in request.form else 0
    guestroom_yes = 1 if 'guestroom_yes' in request.form else 0
    basement_yes = 1 if 'basement_yes' in request.form else 0
    hotwaterheating_yes = 1 if 'hotwaterheating_yes' in request.form else 0
    airconditioning_yes = 1 if 'airconditioning_yes' in request.form else 0
    prefarea_yes = 1 if 'prefarea_yes' in request.form else 0
    furnishingstatus_semi_furnished = 1 if 'furnishingstatus_semi_furnished' in request.form else 0
    furnishingstatus_unfurnished = 1 if 'furnishingstatus_unfurnished' in request.form else 0

    input_features = np.array([[area, bedrooms, bathrooms, stories, parking, mainroad_yes, guestroom_yes, basement_yes,
                                hotwaterheating_yes, airconditioning_yes, prefarea_yes, furnishingstatus_semi_furnished, furnishingstatus_unfurnished]])

    predicted_price = model.predict(input_features)[0]


    return f"The predicted house price is: ${predicted_price:.2f}"

if __name__ == '__main__':
    app.run(debug=True)
