from flask import Flask, request, jsonify
import requests
import numpy as np
import json

app = Flask(__name__)

# Model URLs and weights
models = {
    "participant_1": {"url": "https://f01b-89-30-29-68.ngrok-free.app/predict", "weight": 1.0},
    "participant_2": {"url": "https://7692-89-30-29-68.ngrok-free.app/predict", "weight": 1.0},
    "participant_3": {"url": "https://6b68-89-30-29-68.ngrok-free.app/predict", "weight": 1.0},
    "participant_4": {"url": "https://dedd-89-30-29-68.ngrok-free.app/predict", "weight": 1.0}
}

balances = {name: 1000 for name in models.keys()}  # Initial deposits
slashing_penalty = 100
slashing_threshold = 0.4  # If deviation > 40%, apply penalty

def adjust_weights(consensus, predictions):
    """
    Adjust weights based on accuracy relative to the consensus.
    """
    for name, prediction in predictions.items():
        deviation = np.abs(prediction - consensus).sum() / len(consensus)
        if deviation > slashing_threshold:
            balances[name] -= slashing_penalty
            models[name]["weight"] *= 0.9  # Reduce weight for poor performance
        else:
            models[name]["weight"] *= 1.1  # Reward accurate models

@app.route('/consensus_predict', methods=['GET'])
def consensus_predict():
    try:
        # Gather features from the request
        features = [float(x) for x in request.args.getlist('features')]

        # Collect predictions from all participants
        predictions = {}
        for name, model in models.items():
            response = requests.get(model['url'], params={'features': features}).json()
            probabilities = np.array(response['probabilities'])
            predictions[name] = probabilities * model['weight']

        # Compute consensus prediction (weighted average)
        consensus_probabilities = np.sum(list(predictions.values()), axis=0)
        consensus_prediction = np.argmax(consensus_probabilities)

        # Adjust weights and balances
        adjust_weights(consensus_probabilities, predictions)

        return jsonify({
            'consensus_prediction': int(consensus_prediction),
            'consensus_probabilities': consensus_probabilities.tolist(),
            'balances': balances,
            'weights': {name: model['weight'] for name, model in models.items()}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
