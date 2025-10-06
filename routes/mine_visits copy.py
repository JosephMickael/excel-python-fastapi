from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import ORJSONResponse
import pandas as pd
from datetime import datetime, timedelta
from services.file_utils import df_to_records, normalize_name, read_file, harmonize_columns, unify_name_column, read_topview_file

router = APIRouter()

@router.post("/", response_class=ORJSONResponse)
async def mine_visits(
    file_ifs: UploadFile = File(...),
    file_topview: UploadFile = File(...),
):
    try:
        # === Lecture des fichiers ===
        df_ifs = read_file(file_ifs)  # IFS
        df_topview = read_topview_file(file_topview)  # TopView
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lecture fichier: {e}")

    # === Harmonisation colonnes et unification noms ===
    df_ifs = harmonize_columns(df_ifs)
    df_ifs = unify_name_column(df_ifs)
    df_topview = harmonize_columns(df_topview)
    df_topview = unify_name_column(df_topview)

    # === Normalisation des noms de colonnes TopView ===
    df_topview.columns = (
        df_topview.columns.str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.replace("_", "", regex=False)
        .str.lower()
    )

    # === Vérification colonnes essentielles ===
    expected = ["name", "tagondatetime", "personnelnumber"]
    for col in expected:
        if col not in df_topview.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Le fichier TopView doit contenir une colonne {col}",
            )

    # === Renommage (inclure TagOffDatetime si présent) ===
    rename_map = {
        "name": "Name",
        "tagondatetime": "TagOnDatetime",
        "tagoffdatetime": "TagOffDatetime",
        "personnelnumber": "matricule",
    }
    df_topview.rename(
        columns={k: v for k, v in rename_map.items() if k in df_topview.columns},
        inplace=True,
    )

    # === Extraction Nom / Prénom à partir de Name (si besoin) ===
    if "nom_prenom" not in df_topview.columns:
        parts = df_topview["Name"].astype(str).str.replace("-", " ", regex=False).str.split(" ", n=1, expand=True)
        df_topview["Nom"] = parts[0].fillna("").str.strip().str.upper()
        df_topview["Prénom"] = parts[1].fillna("").str.strip().str.upper() if parts.shape[1] > 1 else ""
        df_topview["nom_prenom"] = (df_topview["Nom"] + " " + df_topview["Prénom"]).str.strip()

    # === Dates ===
    df_topview["TagOnDatetime"] = pd.to_datetime(df_topview["TagOnDatetime"], errors="coerce")
    if "TagOffDatetime" in df_topview.columns:
        df_topview["TagOffDatetime"] = pd.to_datetime(df_topview["TagOffDatetime"], errors="coerce")

    # === Fenêtres ===
    cutoff_date = datetime.now() - timedelta(days=365)
    df_topview_recent = df_topview[df_topview["TagOnDatetime"] >= cutoff_date].copy()
    df_topview_all = df_topview.copy()  # on garde toutes périodes pour les manquants

    # === Normalisation des noms (clé) ===
    df_ifs["nom_prenom"] = df_ifs["nom_prenom"].apply(normalize_name)
    df_topview_recent["nom_prenom"] = df_topview_recent["nom_prenom"].apply(normalize_name)
    df_topview_all["nom_prenom"] = df_topview_all["nom_prenom"].apply(normalize_name)

    # === Clés (matricule|nom_prenom) ===
    df_ifs["key"] = df_ifs["matricule"].astype(str).str.strip() + "|" + df_ifs["nom_prenom"].astype(str).str.strip()
    df_topview_recent["key"] = df_topview_recent["matricule"].astype(str).str.strip() + "|" + df_topview_recent["nom_prenom"].astype(str).str.strip()
    df_topview_all["key"] = df_topview_all["matricule"].astype(str).str.strip() + "|" + df_topview_all["nom_prenom"].astype(str).str.strip()

    # === Manquants dans TopView (inchangé) ===
    # On déduplique à "tous temps" pour lister une seule ligne par personne
    df_topview_all_for_missing_top = df_topview_all.sort_values("TagOnDatetime", ascending=False).drop_duplicates(
        subset=["key"], keep="first"
    )
    ifs_keys = set(df_ifs["key"])
    topview_keys_all = set(df_topview_all_for_missing_top["key"])
    missing_in_topview_all = df_ifs[~df_ifs["key"].isin(topview_keys_all)]

    # === VISITES (inchangé) : on garde TA logique actuelle ===
    merged = pd.merge(
        df_ifs,
        # NOTE: pas de dédoublonnage ici puisque tu voulais garder la logique en place
        df_topview_recent,
        on="key",
        how="inner",
        suffixes=("_ifs", "_topview"),
    )

    id_col_candidates = [c for c in df_ifs.columns if c.strip().lower() in ["id_personne", "id"]]
    id_col = id_col_candidates[0] if id_col_candidates else None
    id_col_merged = f"{id_col}_ifs" if id_col and f"{id_col}_ifs" in merged.columns else id_col

    group_cols = ["key", "nom_prenom_ifs", "matricule_ifs"]
    if "statut_d employe" in merged.columns:
        group_cols.append("statut_d employe")
    if "nom_d organisation" in merged.columns:
        group_cols.append("nom_d organisation")
    if id_col_merged and id_col_merged in merged.columns:
        group_cols.append(id_col_merged)

    visits = (
        merged.groupby(group_cols, dropna=False)
        .agg(
            nb_visites=("TagOnDatetime", "count"),
            derniere_visite=("TagOnDatetime", "max"),
        )
        .reset_index()
    )
    visits["derniere_visite"] = visits["derniere_visite"].apply(
        lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(x) else None
    )

    visites_out = []
    for _, row in visits.iterrows():
        r = {
            "key": row["key"],
            "nom_prenom": row.get("nom_prenom_ifs"),
            "id_personne": row.get(id_col_merged),
            "matricule": row.get("matricule_ifs"),
            "statut": row.get("statut_d employe"),
            "departement": row.get("nom_d organisation"),
            "nb_visites": int(row["nb_visites"]),
            "derniere_visite": row["derniere_visite"],
        }
        visites_out.append(r)

    # === >>> Manquants dans IFS (SEULE PARTIE MODIFIÉE) <<<
    # Utiliser des "sessions" TopView: (TagOnDatetime, TagOffDatetime) sinon fallback par jour
    df_sessions = df_topview_all.copy()
    # (re)calcule la clé à partir de la version normalisée
    df_sessions["nom_prenom"] = df_sessions["nom_prenom"].apply(normalize_name)
    df_sessions["key"] = df_sessions["matricule"].astype(str).str.strip() + "|" + df_sessions["nom_prenom"].astype(str).str.strip()

    has_off = "TagOffDatetime" in df_sessions.columns
    if has_off:
        on_min = df_sessions["TagOnDatetime"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
        off_min = df_sessions["TagOffDatetime"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
        df_sessions["session_id"] = df_sessions["key"] + "|" + on_min + "|" + off_min
    else:
        df_sessions["session_id"] = df_sessions["key"] + "|" + df_sessions["TagOnDatetime"].dt.strftime("%Y-%m-%d").fillna("")

    # Garde uniquement les clés absentes d'IFS
    missing_sessions = df_sessions[~df_sessions["key"].isin(ifs_keys)]

    # Agrège par personne : dernière descente + nb de sessions (nb_visites_miss facultatif)
    missing_in_ifs_all_agg = (
        missing_sessions.groupby(["key", "matricule", "nom_prenom"], dropna=False)
        .agg(
            TagOnDatetime=("TagOnDatetime", "max"),
            nb_visites_miss=("session_id", "nunique"),
        )
        .reset_index()
    )
    # Format date pour JSON et pour ton tableau (colonne TagOnDatetime)
    missing_in_ifs_all_agg["TagOnDatetime"] = missing_in_ifs_all_agg["TagOnDatetime"].apply(
        lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(x) else None
    )

    return ORJSONResponse(
        content={
            "visites": visites_out,  # inchangé
            "missing_in_topview_all": df_to_records(missing_in_topview_all),
            "missing_in_ifs_all": df_to_records(missing_in_ifs_all_agg),
        }
    )
