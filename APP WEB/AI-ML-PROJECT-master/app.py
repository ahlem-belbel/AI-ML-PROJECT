from flask import Flask, render_template
import pickle
import os
from flask import Flask, render_template, request
import numpy as np
app = Flask(__name__)

# Chemin vers le modèle sauvegardé dans le dossier static
model_path = os.path.join(app.root_path, 'static', 'saved_steps.pkl')

# Charger le modèle et les étapes de prétraitement
with open(model_path, 'rb') as model_file:
    saved_data = pickle.load(model_file)

model = saved_data["model"]
gearbox = saved_data["gearbox_encoder"]
bran = saved_data["brand_encoder"]
fuel = saved_data["fuel_encoder"]
carn = saved_data["Car_Name_encoder"]
scaler = saved_data["scaler"]
pca = saved_data["pca"]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/estim.html')  
def estimation():
    return render_template('estim.html')
@app.route('/estimate', methods=['POST'])
def estimate():
    if request.method == 'POST':
        # Récupérer les données du formulaire
        brand = request.form['brand']
        modele=request.form['modele']
        year = int(request.form['year'])
        engine_capacity = float(request.form['engine_capacity'])
        power_value = float(request.form['power_value'])
        mileage_value = float(request.form['mileage_value'])
        gearbox_type = request.form['gearbox_type']
        fuel_type = request.form['fuel_type']

        # Prétraiter les données pour les préparer à être utilisées par le modèle
        # Assurez-vous d'encoder les données catégoriques et de les mettre au format attendu par le modèle
        
        # Encodez les caractéristiques catégoriques
        gearbox_encoded = gearbox.transform([gearbox_type])[0]
        brand_encoded = bran.transform([brand])[0]
        fuel_encoded = fuel.transform([fuel_type])[0]
        model_encoded = carn.transform([modele])[0]

        
        # Créez un dictionnaire de données prétraitées
        new_data = {
            "brand": brand_encoded,
            "Car_Name": model_encoded,  # Vous devez remplacer cela par une valeur appropriée
            "Engine_capcaity1": engine_capacity,
            "Year": year,
            "Power_Value": power_value,
            "Mileage_Value": mileage_value,
            "gearbox_type_english": gearbox_encoded,
            "fuel_type_english": fuel_encoded
        }

        # Convertissez les données en un tableau NumPy
        new_data_array = np.array([list(new_data.values())])

        # Standardisez les données
        new_data_scaled = scaler.transform(new_data_array)

        # Réduisez la dimensionnalité des données
        new_data_reduced = pca.transform(new_data_scaled)

        # Effectuez la prédiction en utilisant le modèle chargé
        predicted_price = model.predict(new_data_reduced)[0]

        # Renvoyez la page HTML avec le résultat de la prédiction
        return render_template('estim.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
