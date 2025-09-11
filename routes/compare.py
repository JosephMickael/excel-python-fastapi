from fastapi import APIRouter, UploadFile, File, HTTPException
from services.file_utils import read_file, harmonize_columns, unify_name_column, compare_files

router = APIRouter()

@router.post("/")
async def compare_endpoint(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    if not file1 or not file2:
        raise HTTPException(status_code=400, detail="Deux fichiers requis")
    try:
        df1 = read_file(file1)
        df2 = read_file(file2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lecture fichier: {e}")

    df1 = harmonize_columns(df1)
    df2 = harmonize_columns(df2)

    df1 = unify_name_column(df1)
    df2 = unify_name_column(df2)

    report = compare_files(df1, df2, key="matricule")
    return {"report": report}
