from fastapi import FastAPI
from routes import compare, harmonize, duplicates
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         
    allow_credentials=True,
    allow_methods=["*"],           
    allow_headers=["*"],           
)

app.include_router(compare.router, prefix="/compare", tags=["compare"])
app.include_router(harmonize.router, prefix="/harmonize", tags=["harmonize"])
app.include_router(duplicates.router, prefix="/duplicates", tags=["duplicates"])

