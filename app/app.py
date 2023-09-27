from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

import __init__

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())

@app.route('/api', methods=["GET", "POST"])
def chatbotResponse():
    if request.method == 'POST':
        response = "Peticion a una api"
    return jsonify({"response": response })

if __name__ == "__main__":
    app.run(port=5000, debug=True)