from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import ORJSONResponse
import pandas as pd
from datetime import datetime, timedelta
from services.file_utils import normalize_name, read_file, harmonize_columns, unify_name_column, read_topview_file

router = APIRouter()

@router.post("/", response_class=ORJSONResponse)
async def mine_visits(
    file_ifs: UploadFile = File(...),
    file_topview: UploadFile = File(...)
):
    try:
        # Lecture des fichiers
        df_ifs = read_file(file_ifs)  # IFS
        df_topview = read_topview_file(file_topview)  # TopView
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lecture fichier: {e}")

    # Harmonisation colonnes et unification noms/prénoms
    df_ifs = harmonize_columns(df_ifs)
    df_ifs = unify_name_column(df_ifs)

    df_topview = harmonize_columns(df_topview)
    df_topview = unify_name_column(df_topview)

    # Normalisation colonnes TopView
    df_topview.columns = (
        df_topview.columns
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.replace("_", "", regex=False)
        .str.lower()
    )

    # Vérifier colonnes essentielles
    if "name" not in df_topview.columns:
        raise HTTPException(status_code=400, detail="Le fichier TopView doit contenir une colonne Name")
    if "tagondatetime" not in df_topview.columns:
        raise HTTPException(status_code=400, detail="Le fichier TopView doit contenir une colonne TagOnDatetime")

    # Renommer pour uniformité
    df_topview.rename(columns={"name": "Name", "tagondatetime": "TagOnDatetime"}, inplace=True)

    # Extraire nom + prénom si nécessaire
    if "nom_prenom" not in df_topview.columns:
        topview_names = df_topview["Name"].str.split("-", n=1, expand=True)
        df_topview["Nom"] = topview_names[0].str.strip().str.upper()
        df_topview["Prénom"] = topview_names[1].str.strip().str.upper()
        df_topview["nom_prenom"] = df_topview["Nom"] + " " + df_topview["Prénom"]

    # Conversion des dates
    df_topview["TagOnDatetime"] = pd.to_datetime(df_topview["TagOnDatetime"], errors="coerce")

    # Filtrer visites sur la dernière année
    cutoff_date = datetime.now() - timedelta(days=365)
    df_topview = df_topview[df_topview["TagOnDatetime"] >= cutoff_date]

    # Normalisation des noms
    df_ifs["nom_prenom"] = df_ifs["nom_prenom"].apply(normalize_name)
    df_topview["nom_prenom"] = df_topview["nom_prenom"].apply(normalize_name)

    # Merge (croisement)
    merged = pd.merge(
        df_ifs,
        df_topview,
        how="inner",
        on="nom_prenom"
    )

    # Détection de la colonne ID dans IFS
    id_col_candidates = [c for c in df_ifs.columns if c.strip().lower() in ["id_personne", "id"]]
    id_col = id_col_candidates[0] if id_col_candidates else None

    # Colonnes de regroupement
    group_cols = ["nom_prenom", "matricule", "statut_d employe", "nom_d organisation"]
    if id_col:
        group_cols.append(id_col)

    # Regroupement
    visits = (
        merged.groupby(group_cols, dropna=False)
        .agg(
            nb_visites=("TagOnDatetime", "count"),
            derniere_visite=("TagOnDatetime", "max")
        )
        .reset_index()
    )

    # Transformer en JSON
    result = []
    for _, row in visits.iterrows():
        r = {
            "nom_prenom": row["nom_prenom"],
            "id_personne": row[id_col] if id_col and id_col in row else None,
            "matricule": row["matricule"],
            "statut": row["statut_d employe"],
            "departement": row["nom_d organisation"],
            "nb_visites": int(row["nb_visites"]),
            "derniere_visite": row["derniere_visite"].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(row["derniere_visite"]) else None
        }
        result.append(r)

    return ORJSONResponse(content={"visites": result})
