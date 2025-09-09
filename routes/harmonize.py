from fastapi import APIRouter, UploadFile, File
from services.file_utils import read_file, harmonize_columns, unify_name_column, safe_value

router = APIRouter()

@router.post("/harmonize")
async def harmonize(file: UploadFile = File(...)):
    df = read_file(file)
    df = harmonize_columns(df)
    df = unify_name_column(df)
    
    # Convertir toutes les valeurs pour JSON
    data = [{col: safe_value(val) for col, val in row.items()} for row in df.to_dict(orient="records")]
    return {"data": data}
