from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.database import engine
from backend import models
from backend.routers import admin, authentication, technician
from backend.core.hashing import Hash

app = FastAPI()
models.Base.metadata.create_all(bind=engine)

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(authentication.router)
app.include_router(admin.router)
app.include_router(technician.router)