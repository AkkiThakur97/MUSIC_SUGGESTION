from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app) # Open communication with local web interfaces

print("Loading Machine Learning Emotion Model...")
# Loading model pipeline securely
emotion_classifier = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base"
)
print("Model loaded successfully!")

# Local phrase checker mappings
TAG_FALLBACKS = {
    "rainy day lofi chill": "sadness",
    "super happy hyped energy": "joy",
    "sad break up crying night vibes": "sadness",
    "scared worried dark night background": "fear"
}

# Standardize output labels matching frontend styling states
MOOD_MAPPING = {
    "joy": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "fear": "fear",
    "surprise": "joy",
    "disgust": "anger",
    "neutral": "neutral"
}

@app.route('/predict-mood', methods=['POST'])
def predict_mood():
    data = request.json or {}
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"vibe": "neutral"})
    
    # Exact phrase priority mapping
    clean_text = text.lower().strip()
    if clean_text in TAG_FALLBACKS:
        return jsonify({"vibe": TAG_FALLBACKS[clean_text]})
        
    try:
        outputs = emotion_classifier(text)
        # Extract model label and cast to lower case to eliminate parsing errors
        model_label = str(outputs[0]['label']).lower()
        
        detected_vibe = MOOD_MAPPING.get(model_label, "neutral")
        return jsonify({"vibe": detected_vibe})
        
    except Exception as e:
        print(f"Prediction error caught: {e}")
        return jsonify({"vibe": "neutral"})

if __name__ == '__main__':
    app.run(port=5000, debug=True)