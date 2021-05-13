from flask import Flask
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):

    def get(self):
        return {
            "data": "Hello World"
        }

api.add_resource(HelloWorld, "/hello")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', threaded=False)
