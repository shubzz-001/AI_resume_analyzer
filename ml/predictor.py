
import pickle

with open("ml/job_role_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_job_role(cleaned_resume_text) :
    prediction = model.predict([cleaned_resume_text])[0]
    confidence = max(model.predict_proba([cleaned_resume_text])[0])

    return prediction, round(confidence * 100, 2)