from flask import Flask, request, jsonify
import distilbert_model as model

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Server is Working'

@app.route('/get_sentiment', methods=['POST'])
def get_sentiment():
    tx = request.get_json(force=True)
    text = tx['Review']
    sent = model.get_prediction(text)
    return jsonify(result = sent)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False, use_reloader = False)