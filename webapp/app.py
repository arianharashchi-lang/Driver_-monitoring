from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DRIVER_PAGE = BASE_DIR / "templates" / "driver.html"

app = FastAPI(title="Driver Monitoring", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse(url="/driver", status_code=307)


@app.get("/driver", response_class=FileResponse)
def driver_page() -> FileResponse:
    return FileResponse(DRIVER_PAGE, media_type="text/html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}