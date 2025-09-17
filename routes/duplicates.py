from fastapi import APIRouter, UploadFile, File, HTTPException
from services.file_utils import read_file, unify_name_column
from services.duplicates_utils import harmonize_columns, normalize_column
import numpy as np
from fastapi.encoders import jsonable_encoder
import pandas as pd

router = APIRouter()

# Colonnes sur lesquelles on veut détecter les doublons
DUPLICATE_COLUMNS = ["matricule", "nom_prenom", "date_naissance"]

@router.post("/")
async def detect_duplicates(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Fichier requis")
    try:
        df = read_file(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lecture fichier: {e}")

    # Harmoniser et normaliser les colonnes
    df = harmonize_columns(df)
    df = unify_name_column(df)

    # Remplacer NaN par None
    df = df.replace({np.nan: None})

    # Vérifier colonnes utiles
    cols_to_check = [col for col in DUPLICATE_COLUMNS if col in df.columns]
    if not cols_to_check:
        raise HTTPException(
            status_code=400,
            detail="Aucune colonne pour détection de doublons trouvée"
        )

    # Extraire les doublons
    duplicates = df[df.duplicated(subset=cols_to_check, keep=False)]
    duplicates = duplicates.replace({np.nan: None})

    result = []

    # Groupement par colonnes clés
    grouped = duplicates.groupby(cols_to_check)

    for keys, group in grouped:
        group = group.replace({np.nan: None})
        row_data = dict(zip(cols_to_check, keys if isinstance(keys, tuple) else [keys]))

        # Extraire le matricule principal
        matricule_general = row_data.get("matricule")

        # Construire la liste des doublons sans matricule
        doublons_sans_matricule = (
            group.drop(columns=["matricule"], errors="ignore")
                 .to_dict(orient="records")
        )

        # Construire l'objet final
        result.append({
            "matricule": matricule_general,
            **{k: v for k, v in row_data.items() if k != "matricule"},
            "nombre_doublons": len(group),
            "doublons": doublons_sans_matricule
        })

    # Trier pour un rendu lisible
    result = sorted(
        result,
        key=lambda x: (x.get("nom_prenom") or "", x.get("matricule") or "")
    )

    return jsonable_encoder({"duplicates": result})

