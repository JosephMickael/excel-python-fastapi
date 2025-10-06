import pandas as pd
import numpy as np


# ==============================================
# === Harmonisation IFS ========================
# ==============================================
def harmonize_ifs_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rename_map = {
        "id_empl.": "matricule",
        "id_personne": "id_personne",
        "prénom": "prenom",
        "deuxième_prénom": "deuxieme_prenom",
        "nom": "nom",
        "statut_d'employé": "statut",
        "nom_d'organisation": "departement",
        "nom_de_la_structure": "departement",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # Créer nom complet
    if "nom" in df.columns and "prenom" in df.columns:
        df["nom_prenom"] = (
            df["nom"].astype(str).str.strip() + " " + df["prenom"].astype(str).str.strip()
        )
    else:
        df["nom_prenom"] = np.nan

    # Colonnes nécessaires
    for c in ["id_personne", "statut", "departement", "matricule"]:
        if c not in df.columns:
            df[c] = np.nan

    # Clé principale = ID personne
    df["key"] = df["id_personne"].astype(str).str.strip()

    return df


# ==============================================
# === Harmonisation Top View ===================
# ==============================================
def harmonize_topview_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Clé principale = PersonID (même que ID personne dans IFS)
    if "personid" in df.columns:
        df["key"] = df["personid"].astype(str).str.strip()
    else:
        df["key"] = np.nan

    # Colonne matricule (PersonnelNumber)
    if "personnelnumber" in df.columns:
        df["matricule"] = df["personnelnumber"].astype(str).str.strip()
    else:
        df["matricule"] = np.nan

    # Nom complet
    if "name" in df.columns:
        df["nom_prenom"] = df["name"].astype(str).str.strip()
    else:
        df["nom_prenom"] = np.nan

    # Date de visite
    if "tagondatetime" in df.columns:
        df["date_visite"] = pd.to_datetime(df["tagondatetime"], errors="coerce")
    elif "tagoffdatetime" in df.columns:
        df["date_visite"] = pd.to_datetime(df["tagoffdatetime"], errors="coerce")
    else:
        df["date_visite"] = pd.NaT

    return df


def process_mine_visits(file_ifs, file_topview):
    print("[DEBUG] Lecture des fichiers Excel")
    df_ifs = pd.read_excel(file_ifs.file)
    df_topview = pd.read_excel(file_topview.file)

    df_ifs = harmonize_ifs_columns(df_ifs)
    df_topview = harmonize_topview_columns(df_topview)
    print("[DEBUG] Harmonisation terminée")

    # === Normalisation de base ===
    def normalize_text(x):
        if pd.isna(x):
            return ""
        return (
            str(x)
            .strip()
            .lower()
            .replace("é", "e")
            .replace("è", "e")
            .replace("ê", "e")
            .replace("à", "a")
            .replace("ç", "c")
            .replace("  ", " ")
        )

    # === Nettoyage nom / prénom / matricule ===
    for col in ["nom", "prenom", "matricule"]:
        if col not in df_ifs.columns:
            df_ifs[col] = ""
        if col not in df_topview.columns:
            df_topview[col] = ""

    df_ifs["nom"] = df_ifs["nom"].apply(normalize_text)
    df_ifs["prenom"] = df_ifs["prenom"].apply(normalize_text)
    df_ifs["matricule"] = df_ifs["matricule"].astype(str).str.strip().str.replace(r"^0+", "", regex=True)

    df_topview["name"] = df_topview["name"].fillna("").astype(str)
    split_names = df_topview["name"].str.split(" ", n=1, expand=True)
    df_topview["nom"] = split_names[0].apply(normalize_text)
    df_topview["prenom"] = split_names[1].apply(normalize_text) if split_names.shape[1] > 1 else ""
    df_topview["matricule"] = df_topview["personnelnumber"].astype(str).str.strip().str.replace(r"^0+", "", regex=True)

    # === Clé de correspondance nom+prenom+matricule ===
    df_ifs["key"] = (df_ifs["nom"] + "_" + df_ifs["prenom"] + "_" + df_ifs["matricule"]).str.strip().str.lower()
    df_topview["key"] = (df_topview["nom"] + "_" + df_topview["prenom"] + "_" + df_topview["matricule"]).str.strip().str.lower()

    # === Calcul des visites ===
    visits = (
        df_topview.groupby("key")
        .agg(nb_visites=("key", "count"), derniere_visite=("date_visite", "max"))
        .reset_index()
    )

    # Conversion propre des dates
    visits["derniere_visite"] = visits["derniere_visite"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # === Fusion avec IFS (nom+prenom+matricule) ===
    extra_cols = [
        "id_personne", "statut", "departement", "nom_prenom", "matricule"
    ]
    cols_existantes = [c for c in extra_cols if c in df_ifs.columns]
    infos_ifs = df_ifs[["key"] + cols_existantes].drop_duplicates("key")

    visits = visits.merge(infos_ifs, on="key", how="left")

    # Ajouter Department du TopView si dispo
    if "department" in df_topview.columns:
        dept_info = df_topview[["key", "department"]].drop_duplicates("key")
        visits = visits.merge(dept_info, on="key", how="left", suffixes=("", "_topview"))

    visits = visits.loc[:, ~visits.columns.duplicated()]

    # === Déterminer les manquants ===
    keys_ifs = set(df_ifs["key"].dropna())
    keys_top = set(df_topview["key"].dropna())

    missing_in_topview = df_ifs[df_ifs["key"].isin(keys_ifs - keys_top)].reset_index(drop=True)
    missing_in_ifs = df_topview[df_topview["key"].isin(keys_top - keys_ifs)].reset_index(drop=True)

    print(f"[DEBUG] Nb manquants dans TopView : {len(missing_in_topview)}")
    print(f"[DEBUG] Nb manquants dans IFS : {len(missing_in_ifs)}")
    print("[DEBUG] Traitement terminé avec succès")

    # === Conversion finale (dates → str pour JSON) ===
    def safe_serialize(df):
        for col in df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
            df[col] = df[col].astype(str)
        return df.to_dict(orient="records")

    return {
        "visites": safe_serialize(visits),
        "missing_in_topview": safe_serialize(missing_in_topview),
        "missing_in_ifs": safe_serialize(missing_in_ifs),
    }
