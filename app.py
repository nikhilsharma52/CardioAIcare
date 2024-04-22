import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    if (prediction[0] == 0):
        str = 'The person doesnt have heart disease'
    else:
        str = 'The person has heart disease'
    return render_template("result.html", prediction=prediction)

@flask_app.route("/reload", methods=["POST"])
def reload():
    return render_template("index.html")

if __name__ == "__main__":
    flask_app.run(debug=True)
