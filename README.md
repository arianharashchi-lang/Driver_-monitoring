# Driver Monitoring

An experimental browser-based dashboard for monitoring observable driver cues and testing pedestrian detection. It displays eye closure, head turns, relative gaze, mouth opening, and optional phone and pedestrian detections.

[Open the driver dashboard](https://facialemotionai-web.onrender.com/driver) · [Run locally](#run-locally)

Project pages: [DIDBAN ایران](docs/didban-iran.html) · [DIDBAN industrial pilot](docs/didban-industrial-pilot.html)

## What it does

- Processes camera frames in the browser for face landmarks, relative gaze, and observable fatigue or distraction cues.
- Offers optional browser-side phone-object detection and pedestrian detection.
- Shows a simulated ignition state and local session alerts.
- Exports a local HTML report.

This is an experimental prototype. It is not safety-certified and cannot reliably assess a person's internal state or true alertness. Camera landmarks, gaze, and object detections can be wrong.

## Run locally

Create and activate a virtual environment, then run:

    pip install -r requirements.txt
    uvicorn webapp.app:app --reload

Open http://127.0.0.1:8000/driver. Camera access requires HTTPS or localhost. Browser models load from public CDNs the first time they are used.

## Repository layout

- webapp/: FastAPI server and driver dashboard
- docs/: project pages
- tests/js/: tests for driver metrics and analytics

## Privacy

Camera processing runs in the browser. The app does not upload camera frames to the server. Session alerts and exported reports stay in the browser unless the user chooses to share them.