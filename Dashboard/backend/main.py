from fastapi import FastAPI, BackgroundTasks
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import threading
import os
import json
from extract import background_capture
from registration import router as register_router
from login import router as login_router
from convert import get_predictions
from fastapi.middleware.cors import CORSMiddleware
from model_retrain import retrain_model, manual_retrain
import logging


logging.basicConfig(level=logging.INFO)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(register_router)
app.include_router(login_router)

@app.on_event("startup")
def start_capture():
    thread = threading.Thread(target=background_capture, daemon=True)
    thread.start()
    logging.info("Background packet capture thread started")

    # Scheduler for retrain
    scheduler = BackgroundScheduler()

    # Calculate delay until next 00:00 (or you can set your preferred time)
    now = datetime.now()
    next_run = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    delay_seconds = (next_run - now).total_seconds()

    # Schedule retrain job daily at midnight
    def scheduled_retrain_wrapper():
        logging.info("Scheduled retraining triggered")
        retrain_model()

    scheduler.add_job(scheduled_retrain_wrapper, 'interval', days=1, next_run_time=next_run)
    scheduler.start()

    logging.info(f"Retraining scheduler started; first run at {next_run}")

@app.get("/")
def read_root():
    return {"message": "FastAPI is working and capturing packets."}

@app.get("/packets")
def get_packets():
    JSON_FILE = "capture_data.json"
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return []

@app.get("/predictions")
def get_predictions_route(page: int = 1, page_size: int = 500):
    predictions = get_predictions(page, page_size)
    if "error" in predictions:
        return {"error": predictions["error"]}
    return predictions


@app.post("/manual_retrain")
def manual_retrain_endpoint(background_tasks: BackgroundTasks):
    background_tasks.add_task(manual_retrain)
    logging.info("Manual retrain endpoint called - retraining started in background")
    return {"message": "Manual retraining started in background"}
