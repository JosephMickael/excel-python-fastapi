from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io

app = FastAPI()

# Mapping des colonnes
MAPPING = {
    "statut": ["statut", "Statut du travail"],
    "matricule": ["matricule", "ID (Matricule Num)", "ID empl.", "ID employé"],
    "sexe": ["sexe", "Sexe"],
    "id_personne": ["id personne", "Person ID"],
    "nom_prenom": ["nom & prenom", "Nom utilisateur", "Nom de famille", "Prénom", "Nom", "Noms", "Prénoms", "Prénom"],
    "date_naissance": ["date naissance", "Date de naissance"],
    "date_emploi": ["date emploi", "Date de début"],
    "date_fin_emploi": ["date fin emploi", "Date de fin"],
    "departement": ["departement", "Département"],
    "poste": ["titre poste", "Poste"],
    "sous_departement": ["sous departement", "Sous-département"],
}

def read_file(uploaded_file: UploadFile) -> pd.DataFrame:
    """Lit un fichier CSV ou Excel et retourne un DataFrame"""
    try:
        if uploaded_file.filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file.file, dtype=str)
        elif uploaded_file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file.file, dtype=str)
        else:
            raise HTTPException(status_code=400, detail="Format non supporté (uniquement CSV ou Excel).")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de lecture fichier : {str(e)}")
    return df

import logging

logger = logging.getLogger("uvicorn.error")

def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes du DataFrame selon le MAPPING et fusionne si besoin"""
    rename_dict = {}
    for key, possible_names in MAPPING.items():
        for col in df.columns:
            col_norm = col.strip().lower()
            if col_norm in [x.lower() for x in possible_names]:
                rename_dict[col] = key
    df = df.rename(columns=rename_dict)

    # Debug : voir les colonnes après mapping
    logger.info(f"Colonnes après mapping : {list(df.columns)}")

    # Si nom_prenom n’existe pas, essayer de le recréer
    if "nom_prenom" not in df.columns:
        if "nom_de_famille" in df.columns and "prénom" in df.columns:
            df["nom_prenom"] = (
                df["prénom"].fillna("").astype(str).str.strip() + " " +
                df["nom_de_famille"].fillna("").astype(str).str.strip()
            ).str.strip()
        elif "nom_de_famille" in df.columns:
            df["nom_prenom"] = df["nom_de_famille"]
        elif "prénom" in df.columns:
            df["nom_prenom"] = df["prénom"]
        else:
            candidates = [c for c in df.columns if "nom" in c.lower() or "pren" in c.lower()]
            if candidates:
                df["nom_prenom"] = df[candidates].astype(str).agg(" ".join, axis=1).str.strip()
            else:
                logger.warning("Aucune colonne nom/prénom détectée, création d'une colonne vide.")
                df["nom_prenom"] = ""

    return df
