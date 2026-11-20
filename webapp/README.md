# Driver monitoring web app

This package serves the browser-based driver dashboard at /driver. Camera and model processing happen in the browser; FastAPI serves the page and static assets.

Run from the repository root:

    pip install -r requirements.txt
    uvicorn webapp.app:app --reload

Open http://127.0.0.1:8000/driver. Camera access requires HTTPS or localhost. Face landmark and object-detection models load in the browser from public CDNs when enabled.