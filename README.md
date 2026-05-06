# EmoSense — Emotion Detection Web App

A Flask + BiLSTM web application that detects emotions from text.

## Project Structure

```
emotion_app/
├── app.py                        ← Flask backend
├── requirements.txt              ← Python dependencies
├── static/
│   └── index.html                ← Frontend UI
├── emotion_bilstm_model.h5       ← (generated after training)
├── text_tokenizer.pkl            ← (generated after training)
├── label_encoder.pkl             ← (generated after training)
└── max_len.pkl                   ← (generated after training)
```

---

## Setup & Run (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train and save your model
Run your Jupyter notebook (`Emotion_detection_using_text.ipynb`) fully.
This will generate 4 files: `emotion_bilstm_model.h5`, `text_tokenizer.pkl`,
`label_encoder.pkl`, and `max_len.pkl`.

Copy all 4 files into the `emotion_app/` folder.

### 3. Enable real predictions in app.py
Open `app.py` and **uncomment** the model loading block at the top
and the real prediction block inside the `/predict` route.
Also **comment out** or delete the DEMO MODE block.

### 4. Run the app
```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## Deploy for College Submission

### Option A — Run locally and demo live (simplest)
Just run `python app.py` on your laptop and demo in the browser.
No hosting needed for an in-person submission.

### Option B — Deploy on Render (free, online URL)
1. Push your project to a GitHub repo
2. Go to https://render.com and sign up free
3. Click **New → Web Service** → connect your GitHub repo
4. Set:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `python app.py`
5. Click Deploy — you get a public URL like `https://emosense.onrender.com`

### Option C — Deploy on Railway (free, easiest)
1. Go to https://railway.app
2. Connect GitHub repo
3. Railway auto-detects Flask and deploys
4. Get a public URL instantly

### Option D — Share as a ZIP (offline submission)
Zip the entire `emotion_app/` folder including all `.pkl` and `.h5` files.
The evaluator runs `pip install -r requirements.txt` then `python app.py`.

---

## Notes
- The app runs in **demo mode** (random predictions) until you copy your
  trained model files in and uncomment the real prediction code in `app.py`.
- Make sure all 4 model files are in the same folder as `app.py`.
