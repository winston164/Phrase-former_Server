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

class ParaphraseApi(Resource):

    def get(self):
        return {
            "data": "Hello World"
        }

    def put(self):
        args = paraphrasing_args.parse_args()
        sentence = args["sentence"]
        diversity = args["diversity"]

        t0 = time.time()
        if(sentence != None and type(sentence) == str):
            if(diversity != None and type(diversity) == float and diversity > 0 ):
                variations = getParaphrases(sentence, diversity)
            else:
                variations = getParaphrases(sentence)

            t1 = time.time()
            print(t1-t0)
            return {"data": variations}
        return {"args": args}

api.add_resource(ParaphraseApi, "/paraphrase")

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', threaded=False)
