from fastapi import FastAPI
from routes import compare, harmonize, duplicates
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Connexion frontend angular
origins = [
    "http://localhost:4200",
    "https://root2rise-nsc-formation-dev.ca",
    "http://208.109.241.173",
    "http://208.109.231.28",
    "http://www.root2rise-nsc-formation-dev.ca"
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

