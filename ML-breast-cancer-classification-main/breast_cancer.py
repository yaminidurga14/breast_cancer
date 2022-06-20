from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
loaded_model = pickle.load(open("log_reg.pkl", "rb"))


@app.route("/")
def home():
    return render_template("form.html")


@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    feature_values = [np.array(input_features)]

    feature_names = ['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean',
                     'concavity_mean', 'symmetry_mean', 'fractal_dimension_mean',
                     'radius_se', 'texture_se', 'smoothness_se', 'compactness_se',
                     'concavity_se', 'concave points_se', 'symmetry_se',
                     'fractal_dimension_se', 'smoothness_worst', 'compactness_worst',
                     'concavity_worst', 'symmetry_worst', 'fractal_dimension_worst',
                     ]

    df = pd.DataFrame(feature_values, columns=feature_names)
    output = loaded_model.predict(df)

    if output == 1:
        val = "Malignant(Cancerous)"
    else:
        val = "Benign(Non-Cancerous)"

    return render_template('form.html', prediction_text='Patient has a {} tumor.'.format(val))


if __name__ == "__main__":
    app.run()
