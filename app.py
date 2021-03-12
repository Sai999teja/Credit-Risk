from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import joblib

app = Flask(__name__)

from sklearn.ensemble import GradientBoostingClassifier

model = pickle.load(open('credit_model1.pkl','rb'))



@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('index.html',pred='There is a high chance of defaulting your loan, based on the entered data.')
    else:
        return render_template('index.html',pred='we can santion him a loan, based on the entered data.')

if __name__ == '__main__':
    app.run(debug=True)
