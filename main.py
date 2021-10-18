import json
import sys
from prediction.evaluate_model import Evaluate
from rest_api.RestApi import PredictTopic

from flask import Flask
from flask_restful import Api

data = json.loads(sys.argv[1])

print(data)

app = Flask(__name__)

@app.route('/topic',methods =["GET"])
def topic():
    return PredictTopic.get(data['statement'])

app.run(host="0.0.0.0",port=4444, debug=True)


