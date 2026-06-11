from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db import init_db
from routes.delivery import router as delivery_router
from routes.c2c import router as c2c_router
from routes.pickups import router as pickups_router
from routes.confirmation import router as confirmation_router
from routes.warehouse import router as warehouse_router
from routes.admin import router as admin_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()  # ensure MongoDB indexes exist on startup
    yield


app = FastAPI(title="Chonchoun Courier API", lifespan=lifespan)

# Allow browser clients (Flutter web admin) to call the API cross-origin.
# Open in all environments since the API is public read; tighten allow_origins
# to specific hosts if auth-bearing endpoints are added later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(delivery_router)
app.include_router(c2c_router)
app.include_router(pickups_router)
app.include_router(confirmation_router)
app.include_router(warehouse_router)
app.include_router(admin_router)
