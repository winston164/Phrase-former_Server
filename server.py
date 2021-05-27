from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
from nlp_model_english import getParaphrases
import time

app = Flask(__name__)
cors = CORS(app, methods=["GET","PUT"], resources={r"/*": {"origins": "*"}})
api = Api(app)

paraphrasing_args = reqparse.RequestParser()
paraphrasing_args.add_argument("sentence", type=str, help="The sentence to be paraphrased is required" )
paraphrasing_args.add_argument("diversity", type=float, help="The sentence to be paraphrased")
paraphrasing_args.add_argument("exclude",  help="Words to be excluded from results")

class ParaphraseApi(Resource):

    def get(self):
        return {
            "data": "Hello World"
        }

    def put(self):
        args = paraphrasing_args.parse_args()
        sentence = args["sentence"]
        diversity = args["diversity"]
        exclude = args["exclude"]

        if(sentence == None or type(sentence) != str):
            return {"args": args}

        diversity = diversity if diversity != None and type(diversity) == float and diversity > 0 else 0.1
        exclude = exclude if exclude != None and type(exclude) == list else []

        exclude = list(filter(lambda x: type(x) == str, exclude))
        
        variations = getParaphrases(sentence, diversity, exclude)

        return {"data": variations}

api.add_resource(ParaphraseApi, "/paraphrase")

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', threaded=False)
