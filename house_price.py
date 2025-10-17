import numpy as np
import pandas as pd
import joblib
from flask import Flask,request,jsonify


model = joblib.load("house_price_dummy.pkl")

app = Flask(__name__)

@app.route("/",methods = ["GET"])
def start():
  return "hello"


@app.route("/greet",methods =["GET"])
def write():
   return "welcome to my page"


@app.route("/predict",methods = ["POST"])
def begin():
  try:
    data = request.get_json()

    main_data = pd.DataFrame([data],columns = ["House_size","Bedrooms","Bathrooms","Location",	"Furnishing"])

    predictions = model.predict(main_data)

    result = np.array(predictions).tolist()

    return jsonify(result[0])
 
  except Exception as e:
    return jsonify({"error":str(e)})





if __name__ =="__main__":
  app.run(debug=True)

