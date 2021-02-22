
import flask
import numpy as np
import pandas as pd
from pycaret.classification import *
import connexion

model = None

# app = connexion.App(__name__, specification_dir='./')
# app.add_api('swagger.yml')

app = flask.Flask(__name__)

def load():
    global model
    model = load_model('./lgbm')

@app.route("/predict", methods=["POST"])
def predict():
    response = {
        "success": False,
        "Content-Type": "application/json"
    }

    if flask.request.method == "POST":
        if flask.request.get_json().get("feature"):
            # read feature from json and convert to dataframe
            features = flask.request.get_json().get("feature")
            df_X = pd.DataFrame.from_dict(features)
            df_X.replace('', np.nan, inplace=True)
            # predict
            response["prediction"] = str(predict_model(model, data=df_X)['Label'][0])
            # indicate that the request was a success
            response["success"] = True
    # return the data dictionary as a JSON response
    return flask.jsonify(response)

if __name__ == "__main__":
    load()
    print("Server is running ...")
    app.run(host='0.0.0.0', port=5000, debug=True)