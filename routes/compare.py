from fastapi import APIRouter, UploadFile, File, HTTPException
from services.file_utils import clean_report, ensure_key, read_file, harmonize_columns, unify_name_column, compare_files
from typing import Optional
from fastapi.responses import ORJSONResponse

router = APIRouter()

@router.post("/", response_class=ORJSONResponse)
async def compare_endpoint(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    file3: Optional[UploadFile] = File(None)  
):
    if not file1 or not file2:
        raise HTTPException(status_code=400, detail="Deux fichiers minimum sont requis")

    try:
        df1 = read_file(file1)
        df2 = read_file(file2)
        df3 = read_file(file3) if file3 else None
    except Exception as e:
        print("Erreur backend:", e)
        raise HTTPException(status_code=400, detail=f"Erreur lecture fichier: {e}")

    # Harmonisation des colonnes
    df1 = harmonize_columns(df1)
    df2 = harmonize_columns(df2)
    if df3 is not None:
        df3 = harmonize_columns(df3)

    # Unification des noms/prénoms
    df1 = unify_name_column(df1)
    df2 = unify_name_column(df2)
    if df3 is not None:
        df3 = unify_name_column(df3)
        
    ensure_key(df1, "matricule", "df1")
    ensure_key(df2, "matricule", "df2")
    if df3 is not None:
        ensure_key(df3, "matricule", "df3")

        # Comparaison
    report = compare_files(df1, df2, df3=df3, key="matricule")

    return ORJSONResponse(content={"report": report})