# Importing libraries
import pandas as pd
import numpy as np
import joblib
import warnings

# Dictionary với key là tiếng Anh và value là tiếng Việt
word_dict = {
    'Fungal infection': 'Nhiễm nấm',
    'Allergy': 'Dị ứng',
    'GERD': 'GERD',
    'Chronic cholestasis': 'Xơ gan mạn tính',
    'Drug Reaction': 'Phản ứng thuốc',
    'Peptic ulcer diseae': 'Bệnh loét dạ dày',
    'AIDS': 'AIDS',
    'Diabetes ': 'Tiểu đường',
    'Gastroenteritis': 'Viêm dạ dày',
    'Bronchial Asthma': 'Hen suyễn',
    'Hypertension ': 'Huyết áp cao',
    'Migraine': 'Chứng đau nửa đầu',
    'Cervical spondylosis': 'Thoái hóa cột sống cổ',
    'Paralysis (brain hemorrhage)': 'Tê liệt (chảy máu não)',
    'Jaundice': 'Jaundice',
    'Malaria': 'Sốt rét',
    'Chicken pox': 'Thủy đậu',
    'Dengue': 'Sốt xuất huyết',
    'Typhoid': 'Thương hàn',
    'hepatitis A': 'Viêm gan A',
    'Hepatitis B': 'Viêm gan B',
    'Hepatitis C': 'Viêm gan C',
    'Hepatitis D': 'Viêm gan D',
    'Hepatitis E': 'Viêm gan E',
    'Alcoholic hepatitis': 'Viêm gan do rượu',
    'Tuberculosis': 'Lao',
    'Common Cold': 'Cảm lạnh thông thường',
    'Pneumonia': 'Viêm phổi',
    'Dimorphic hemmorhoids(piles)': 'Trĩ đa dạng (trĩ nứt)',
    'Heart attack': 'Trái tim đau',
    'Varicose veins': 'Uy tĩnh mạch',
    'Hypothyroidism': 'Thấp nang tuyến giáp',
    'Hyperthyroidism': 'Tăng hoạt động tuyến giáp',
    'Hypoglycemia': 'Đái tháo đường thấp',
    'Osteoarthristis': 'Thoái hóa khớp',
    'Arthritis': 'Viêm khớp',
    '(vertigo) Paroymsal  Positional Vertigo': '(chói lọi) Chói loạng tư thế địa vị',
    'Acne': 'Mụn trứng cá',
    'Urinary tract infection': 'Nhiễm trùng đường tiểu',
    'Psoriasis': 'Vảy nến',
    'Impetigo': 'Lang ben'
}



warnings.filterwarnings("ignore", category=UserWarning)

# Loading the trained Random Forest model and related data
model_data = joblib.load('random_forest_model_data.joblib')
loaded_model = model_data['model']
symptoms = model_data['X_columns']
encoder = model_data['encoder']

# Symptoms prediction function
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom_index[value] = index

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

def predictCertainDiseases(symptoms):
    symptoms = symptoms.split(",")

    # Creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        if symptom in data_dict["symptom_index"]:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
        else:
            print(f"Warning: Symptom '{symptom}' not recognized.")

    # Reshaping the input data and converting it
    # into a suitable format for model predictions
    input_data = np.array(input_data).reshape(1, -1)

    # Generating probability outputs for each class
    rf_probabilities = loaded_model.predict_proba(input_data)[0]

    # Checking if any disease has a probability greater than 0.1
    certain_predictions = [
        data_dict["predictions_classes"][i]
        for i, prob in enumerate(rf_probabilities)
        if prob > 0.1
    ]

    if certain_predictions:
        # Printing the list of certain predicted diseases
        print("Certain Predicted Diseases:")
        for disease in certain_predictions:
            print(f"{word_dict[disease]}")
    else:
        print("No certain prediction for any disease.")

# Example of using the updated predictCertainDiseases function
predictCertainDiseases("joint_pain,anxiety")


from flask import Flask, request, jsonify

app = Flask(__name__)

def predictCertainDiseases(symptoms):
    symptoms = symptoms.split(",")

    # Creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        if symptom in data_dict["symptom_index"]:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
        else:
            return jsonify({"error": f"Symptom '{symptom}' not recognized."})

    # Reshaping the input data and converting it
    # into a suitable format for model predictions
    input_data = np.array(input_data).reshape(1, -1)

    # Generating probability outputs for each class
    rf_probabilities = loaded_model.predict_proba(input_data)[0]

    # Checking if any disease has a probability greater than 0.1
    certain_predictions = [
        data_dict["predictions_classes"][i]
        for i, prob in enumerate(rf_probabilities)
        if prob > 0.1
    ]

    if certain_predictions:
        # Returning the list of certain predicted diseases
        return {"certain_predicted_diseases": [disease for disease in certain_predictions]}
    else:
        return {"message": "No certain prediction for any disease."}

@app.route('/predict_diseases', methods=['GET', 'POST'])
def predict_diseases():
    try:
        if request.method == 'GET':
            symptoms = request.args.get('symptoms')
        elif request.method == 'POST':
            data = request.get_json()
            symptoms = data.get('symptoms')
        else:
            return jsonify({"error": "Invalid request method."})

        result = predictCertainDiseases(symptoms)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
