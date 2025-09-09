from fastapi import FastAPI
from routes import compare, harmonize, duplicates

app = FastAPI()

app.include_router(compare.router, prefix="/compare", tags=["compare"])
app.include_router(harmonize.router, prefix="/harmonize", tags=["harmonize"])
app.include_router(duplicates.router, prefix="/duplicates", tags=["duplicates"])