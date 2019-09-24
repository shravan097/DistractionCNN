from flask import Flask,request
from cnn  import CNNModel
import json
import tensorflow as tf
from flask_cors import CORS

tf.enable_eager_execution()
app = Flask(__name__)
model = CNNModel()
CORS(app)

@app.route("/predict",methods=['POST'])
def predict():
	data = json.loads(request.data)
	response = model.predict(data['image'])
	print('Response sent as: ', response)
	return response

if __name__ == "__main__":
	app.run()