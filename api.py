from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from db import init_db
from routes.delivery import router as delivery_router
from routes.c2c import router as c2c_router
from routes.pickups import router as pickups_router
from routes.confirmation import router as confirmation_router

app = FastAPI(
    title="Chonchoun Courier API",
    description="3-leg courier system: last-mile ETA, C2C, and first-mile pickup assignment.",
    version="1.0.0",
)

init_db()  # ensure legacy psycopg2 tables exist on startup

app.include_router(delivery_router, prefix="/api/v1")
app.include_router(c2c_router, prefix="/api/v1")
app.include_router(pickups_router, prefix="/api/v1")
app.include_router(confirmation_router, prefix="/api/v1")
