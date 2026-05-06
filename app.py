from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import os

app = Flask(__name__, static_folder='static')

# ── Load model + artifacts ───────────────────────────────────────────────────
# Uncomment these after you have trained and saved your model files
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import neattext as nt
import joblib

model    = load_model('emotion_bilstm_model.h5')
tokenizer = joblib.load('text_tokenizer.pkl')
encoder   = joblib.load('label_encoder.pkl')
MAX_LEN   = joblib.load('max_len.pkl')

# ── Emotion metadata ─────────────────────────────────────────────────────────
EMOTION_META = {
    "happy":     {"emoji": "😊", "color": "#F59E0B", "desc": "Joyful & Positive"},
    "sad":       {"emoji": "😢", "color": "#60A5FA", "desc": "Melancholic & Down"},
    "angry":     {"emoji": "😠", "color": "#EF4444", "desc": "Frustrated & Upset"},
    "fearful":   {"emoji": "😨", "color": "#8B5CF6", "desc": "Anxious & Scared"},
    "disgusted": {"emoji": "🤢", "color": "#10B981", "desc": "Repulsed & Averse"},
    "surprised": {"emoji": "😲", "color": "#F97316", "desc": "Shocked & Astonished"},
    "neutral":   {"emoji": "😐", "color": "#94A3B8", "desc": "Calm & Indifferent"},
}

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # ── REAL PREDICTION (uncomment when model is ready) ──────────────────────
    import neattext as nt
    cleaned = nt.remove_stopwords(nt.remove_special_characters(text).replace("'", "")).lower()
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, padding='post', maxlen=MAX_LEN)
    probs   = model.predict(padded)[0]
    classes = encoder.classes_
    predicted_emotion = classes[np.argmax(probs)]
    confidence = float(np.max(probs))
    all_probs = {cls: float(p) for cls, p in zip(classes, probs)}


    meta = EMOTION_META[predicted_emotion]
    return jsonify({
        'emotion':    predicted_emotion,
        'confidence': round(confidence * 100, 1),
        'emoji':      meta['emoji'],
        'color':      meta['color'],
        'desc':       meta['desc'],
        'all_probs':  {e: round(p * 100, 1) for e, p in all_probs.items()},
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
