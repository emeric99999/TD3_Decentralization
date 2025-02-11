from flask import Flask, request, jsonify
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import json
import requests

# Charger le dataset Iris
iris = datasets.load_iris()
X = iris.data  # Les caractéristiques
y = iris.target  # Les étiquettes
print(type(X))

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Créer et entraîner un modèle SVM
svm_model = SVC(kernel='linear', C=1.0, probability = True, random_state=42)  # Utilisation d'un kernel linéaire
svm_model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = svm_model.predict(X_test)
y_proba = svm_model.predict_proba(X_test)
# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Afficher les coefficients pour un kernel linéaire (facultatif)
if svm_model.kernel == 'linear':
    print("\nCoefficients (poids des caractéristiques):")
    print(svm_model.coef_)

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['GET'])
def predict():
    try:
        features = [float(x) for x in request.args.getlist('features')]
        if len(features) != 4:
            return jsonify({'error': 'Invalid feature length'}), 400

        probabilities = svm_model.predict_proba([features])[0]
        prediction = np.argmax(probabilities)
        return jsonify({
            'prediction': int(prediction),
            'probabilities': probabilities.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
app.run(host="0.0.0.0")