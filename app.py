from flask import Flask, jsonify
import joblib
import numpy as np

app=Flask(__name__)

model=joblib.load("random_forest.pkl")

@app.route('/predict',methods=['GET'])
def predict():

    feature=[130,89,23.5]

    input=np.array(feature)[1,-1]

    prediction=model.predict(input)[0]

    if prediction ==0:
        return jsonify({"prediction":"non-diabetic"})
    else:
        return jsonify({"prediction":"diabetic"})
    
if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)