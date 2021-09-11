#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('heart_failure.pickle', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if(output==1):
        return render_template('index.html', prediction_text='Have high Risk of heart failure ! ☹')
    else:
        return render_template('index.html', prediction_text='Low Risk of heart failure 😃')

        
   # return render_template('index.html', prediction_text='Risk of heart failure (0 for low risk,1 for high risk) : {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
