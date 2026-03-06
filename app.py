from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Initialize the FastAPI app
app = FastAPI(
    title="AI Disaster Response System",
    description="An API that categorizes emergency messages using Machine Learning."
)

# Load the trained AI model
try:
    model = joblib.load('disaster_model.pkl')
except FileNotFoundError:
    raise RuntimeError("Model file not found. Please run train_model.py first.")

# Define the data structure expected from users
class EmergencyMessage(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "AI Disaster Response API is active. Send POST requests to /predict"}

@app.post("/predict")
def predict_emergency(msg: EmergencyMessage):
    if not msg.text.strip():
        raise HTTPException(status_code=400, detail="Message text cannot be empty.")
    
    # Make a prediction
    prediction = model.predict([msg.text])[0]
    
    # Calculate a mock confidence score (urgency proxy)
    probabilities = model.predict_proba([msg.text])[0]
    confidence = max(probabilities) * 100
    
    # Determine basic urgency based on category
    urgency = "HIGH" if prediction != "None" else "LOW"

    return {
        "original_message": msg.text,
        "assigned_category": prediction,
        "confidence_score": f"{confidence:.2f}%",
        "urgency_level": urgency
    }
