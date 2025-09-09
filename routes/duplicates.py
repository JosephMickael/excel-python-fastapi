from fastapi import APIRouter, UploadFile, File, HTTPException
from services.file_utils import read_file, harmonize_columns, unify_name_column, safe_value
import pandas as pd

router = APIRouter()

# Colonnes sur lesquelles on veut détecter les doublons
DUPLICATE_COLUMNS = ["matricule", "nom_prenom", "id_personne", "date_naissance"]

@router.post("/")
async def detect_duplicates(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Fichier requis")
    try:
        df = read_file(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lecture fichier: {e}")

    df = harmonize_columns(df)
    df = unify_name_column(df)

    # Garde uniquement les colonnes utiles
    cols_to_check = [col for col in DUPLICATE_COLUMNS if col in df.columns]
    if not cols_to_check:
        raise HTTPException(status_code=400, detail="Aucune colonne pour détection de doublons trouvée")

    # Détection des doublons
    duplicates = df[df.duplicated(subset=cols_to_check, keep=False)]

    # Retourne les lignes doublons en JSON compatible
    result = [
        {col: safe_value(row[col]) for col in cols_to_check if col in row}
        for idx, row in duplicates.iterrows()
    ]
    
    return {"duplicates": result}
