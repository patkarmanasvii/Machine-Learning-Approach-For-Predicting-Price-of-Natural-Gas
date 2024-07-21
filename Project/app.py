from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model (ensure 'gas.pkl' is in the same directory)
model = joblib.load('gas.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def y_predict():
    try:
        # Retrieve form data
        year = int(request.form['Year'])
        month = int(request.form['Month'])
        day = int(request.form['Day'])
        
        # Prepare input features for the model
        input_features = np.array([[year, month, day]])
        
        # Predict using the loaded model
        prediction = model.predict(input_features)
        
        # Format the prediction output
        prediction_text = f"Predicted price for {year}-{month:02d}-{day:02d} is ${prediction[0]:.2f}"
        
        return render_template('predict.html', prediction_text=prediction_text)
    except Exception as e:
        return render_template('predict.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True, port=5001)
