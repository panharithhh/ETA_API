from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
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

app.include_router(delivery_router)
app.include_router(c2c_router)
app.include_router(pickups_router)
app.include_router(confirmation_router)
app.include_router(warehouse_router)
app.include_router(admin_router)
