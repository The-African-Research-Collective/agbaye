from contextlib import asynccontextmanager

from fastapi import FastAPI
from mangum import Mangum

from app.api.endpoints import model
from app.domain.model import ModelController


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.lid_model = ModelController()
    yield
    del app.state.lid_model


app = FastAPI(lifespan=lifespan)
app.include_router(model.router, prefix='/model', tags=['model'])


@app.get("/")
def read_root():
    return {"Hello": "Welcome to Dynalab 2.0"}


handler = Mangum(app)
