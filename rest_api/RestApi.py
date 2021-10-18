from flask_restful import reqparse, Resource
from prediction.evaluate_model import Evaluate

parser = reqparse.RequestParser()
parser.add_argument('query')

class PredictTopic(Resource):
        

    def get(text):

        
        evaluate_obj = Evaluate()
        print("---------------Initialization completed---------------")
        result = evaluate_obj.predict(text)
        final_result = {'Text': text, 'topic':result}


        return final_result