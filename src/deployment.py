import pandas as pd
import dill

def predict_stroke_probability(patient_data, model):
    """
    Predict stroke probability for a new patient.

    Args:
        patient_data (dict): Dictionary with patient features.
        model: Trained model (pipeline).

    Returns:
        dict: Stroke probability, binary prediction, and risk level.
    """
    try:
        # Convert input dictionary to DataFrame
        patient_df = pd.DataFrame([patient_data])

        # Predict probability and class
        probability = model.predict_proba(patient_df)[0, 1]
        prediction = model.predict(patient_df)[0]

        return {
            'stroke_probability': float(probability),
            'predicted_stroke': bool(prediction),
            'risk_level': (
                'High' if probability > 0.5 else
                'Medium' if probability > 0.3 else
                'Low'
            )
        }

    except Exception as e:
        return {'error': str(e)}


def save_prediction_function(model, filename="stroke_prediction_function.pkl"):
    """
    Save the stroke prediction function with dill.

    Args:
        model: Trained model (pipeline).
        filename (str): Path to save the serialized function.
    """
    with open(filename, 'wb') as f:
        dill.dump(lambda data: predict_stroke_probability(data, model), f)
    print(f"Prediction function saved as: {filename}")
