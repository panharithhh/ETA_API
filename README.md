# Delivery ETA Predictor

## LaDe Dataset

`LaDe/` is not included in this repo. You need to clone it manually from HuggingFace before training:

```bash
git clone https://huggingface.co/datasets/Cainiao-AI/LaDe
```

Place the cloned `LaDe/` folder in the root of this project. The model expects the delivery CSVs at `LaDe/delivery/delivery_*.csv`.

---

## Configuration

In [model.py](model.py), set `BASE_DIR` to your local project path:

```python
BASE_DIR = "/your/path/to/chonchoun"
```

Create a `.env` file in the project root with your secret API key:

```
API_KEY=your_secret_key_here
```

Generate a secure key with:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

> `.env` is gitignored and never committed.

### Calling from Next.js

Pass the key in the `X-API-Key` header:

```js
const res = await fetch("http://your-server/predict_delivery", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": process.env.ML_API_KEY,
  },
  body: JSON.stringify({ ... }),
});
```

Add `ML_API_KEY=your_secret_key_here` to your Next.js `.env.local`.

---

## Setup

**1. Create the virtual environment**
```bash
python -m venv .venv
```

**2. Activate it**
```bash
# Mac / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

You should see `(.venv)` appear in your terminal prompt. To deactivate later just run `deactivate`.

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Train the model** (generates `model.pkl`)
```bash
python model.py
```

**5. Start the server**
```bash
uvicorn api:app --reload
```

API will be running at `http://127.0.0.1:8000`  
Interactive docs at `http://127.0.0.1:8000/docs`

---

## Predict Endpoint

**POST** `/predict`

### Request
```json
{
  "order_id": 1001,
  "accept_time": "2026-04-29T14:30:00",
  "accept_gps_lng": 104.9282,
  "accept_gps_lat": 11.5564,
  "delivery_gps_lng": 104.9145,
  "delivery_gps_lat": 11.5681
}
```

### Response
```json
{
  "order_id": 1001,
  "predicted_minutes": 18.43,
  "eta": "2026-04-29T14:48:26+07:00"
}
```
