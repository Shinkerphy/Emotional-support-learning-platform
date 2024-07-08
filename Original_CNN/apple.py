import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from backend.firebase_credentials import db  # Import Firebase configuration

app = Flask(__name__)

# Load your pre-trained Keras model
model = load_model('/Users/abdulmalikshinkafi/Emotion-Recognition-App/Basic_CNN/model.h5')

# Define a route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    img_array = np.array(data['image']).reshape(48, 48, 1)  # Ensure the image is the right shape
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    maxindex = int(np.argmax(prediction))
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    emotion = emotion_dict[maxindex]

    # Save emotion to Firebase
    db.collection('emotions').add({
        'timestamp': datetime.utcnow(),
        'emotion': emotion
    })

    return jsonify({'emotion': emotion})

# Endpoint to check for interventions
@app.route('/intervene', methods=['GET'])
def intervene():
    current_time = datetime.utcnow()
    start_time = current_time - timedelta(seconds=30)
    emotions_ref = db.collection('emotions')
    query = emotions_ref.where('timestamp', '>=', start_time)
    emotions = query.stream()
    emotion_counts = {}

    for emotion in emotions:
        emotion_data = emotion.to_dict()
        if emotion_data['emotion'] in emotion_counts:
            emotion_counts[emotion_data['emotion']] += 1
        else:
            emotion_counts[emotion_data['emotion']] = 1

    response = {}
    for emotion, count in emotion_counts.items():
        if count >= 3:  # Assuming roughly 3 samples in 30 seconds
            if emotion == "Angry":
                response[emotion] = "Calm down or take a break."
            elif emotion == "Happy":
                response[emotion] = "That's good! Keep up the good work!"
            # Add more interventions as needed

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)