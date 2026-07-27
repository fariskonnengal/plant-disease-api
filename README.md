# 🌿 Plant Disease Detection API

A production-style REST API that serves a trained deep learning model for plant disease classification — built with FastAPI, deployed live on Render, with automatic interactive documentation.

> Upload a leaf image → get a disease prediction across 38 classes → with a confidence score, in real time, from anywhere.

**🔗 Live API docs:** [https://plant-disease-api-v3oy.onrender.com/docs](https://plant-disease-api-v3oy.onrender.com/docs)
*(Free-tier hosting — the first request after inactivity may take 30-60s to wake up.)*

---

## What This Is

This is the deployment layer for my [Plant Disease Detection](https://github.com/fariskonnengal/plant_disease_detection) project. That project trained and demoed the model in Streamlit; this one takes the same trained model and exposes it as a real, callable REST API — the same pattern used to serve ML models in production at companies, rather than a browser-only demo.

## Features

- **`POST /predict`** — accepts an image upload, returns the predicted disease class and confidence score
- **`GET /health`** — health check endpoint confirming the API and model are running
- **Input validation & error handling** — rejects non-image files and malformed requests with clear error messages instead of crashing
- **Auto-generated interactive docs** (`/docs`) — test the API directly in the browser, no separate tools needed
- **Deployed on Render**, publicly reachable, not just local

## Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI |
| Server | Uvicorn |
| Model | TensorFlow / Keras — MobileNetV2 (transfer learning) |
| Image Processing | Pillow (PIL) |
| Hosting | Render |

## How It Works

1. Client sends a `POST` request to `/predict` with an image file
2. FastAPI validates the upload is actually an image (returns `400` if not)
3. The image is converted to RGB, resized to 224×224, and normalized
4. The pre-loaded MobileNetV2 model runs inference
5. The API returns the predicted class and confidence as JSON

```json
{
  "prediction": "Peach___Bacterial_spot",
  "confidence": 95.68
}
```

## Supported Classes

38 classes across Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Pepper (Bell), Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato — covering both diseased and healthy leaf states.

## Running Locally

```bash
git clone https://github.com/fariskonnengal/plant-disease-api.git
cd plant-disease-api
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

Then visit `http://127.0.0.1:8000/docs`.

## API Reference

### `POST /predict`
**Body:** `multipart/form-data` with key `image` (file)

**Response `200`:**
```json
{ "prediction": "string", "confidence": 0.0 }
```

**Response `400`:** invalid or missing image
```json
{ "detail": "Uploaded file must be an image." }
```

### `GET /health`
**Response `200`:**
```json
{ "status": "ok", "model_loaded": true }
```

## Notes / Known Limitations

- Hosted on Render's free tier — spins down after inactivity, so the first request may be slow
- Model's convolutional base is frozen (transfer learning only), trained on the PlantVillage dataset
