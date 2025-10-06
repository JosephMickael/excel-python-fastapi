import pandas as pd
import unicodedata
from rapidfuzz import fuzz
from services.file_utils import read_file, read_topview_file, normalize_name
import time


def strip_accents(text: str):
    """Supprime les accents d'une chaîne Unicode."""
    if not isinstance(text, str):
        return text
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


def normalize_colnames(df):
    """Nettoie les noms de colonnes : minuscules, sans accents, espaces ni ponctuation."""
    df.columns = (
        df.columns
        .map(strip_accents)     
        .str.strip()
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace("_", "", regex=False)
        .str.replace("'", "", regex=False)
    )
    return df

# ALGORITHME DE LEVENSHTEIN
def find_doubtful_matches_main(file_ifs, file_topview, min_score=70, max_score=95):
    start_time = time.time()
    print("=== [DEBUG] Lecture des fichiers ===")
    df_ifs = read_file(file_ifs)
    df_tv = read_topview_file(file_topview)
    print(f"[DEBUG] IFS: {len(df_ifs)} lignes chargées")
    print(f"[DEBUG] TopView: {len(df_tv)} lignes chargées")

    # Normalisation des colonnes
    df_ifs = normalize_colnames(df_ifs)
    df_tv = normalize_colnames(df_tv)
    print("[DEBUG] Colonnes IFS:", list(df_ifs.columns))
    print("[DEBUG] Colonnes TopView:", list(df_tv.columns))

    # Détection des colonnes principales
    id_col_ifs = next((c for c in df_ifs.columns if "idempl" in c), None)
    nom_col = next((c for c in df_ifs.columns if c == "nom"), None)
    prenom_col = next((c for c in df_ifs.columns if "prenom" in c), None) 
    id_col_tv = next((c for c in df_tv.columns if "personnelnumber" in c), None)
    name_col_tv = next((c for c in df_tv.columns if c in ["name", "nomprenom", "fullname"]), None)

    if not id_col_ifs:
        raise ValueError("Colonne matricule non trouvée dans IFS (ex: 'ID empl.')")
    if not nom_col or not prenom_col:
        raise ValueError("Colonnes 'Nom' et 'Prénom' manquantes dans IFS après normalisation")
    if not id_col_tv or not name_col_tv:
        raise ValueError("Colonnes 'PersonnelNumber' ou 'Name' manquantes dans TopView")

    print(f"[DEBUG] ID IFS = {id_col_ifs}, Nom IFS = {nom_col}, Prénom IFS = {prenom_col}")
    print(f"[DEBUG] ID TV = {id_col_tv}, Nom TV = {name_col_tv}")

    # Création du champ nom_prenom
    df_ifs["nom_prenom"] = (
        df_ifs[nom_col].astype(str) + " " + df_ifs[prenom_col].astype(str)
    ).map(normalize_name)
    df_tv["nom_prenom"] = df_tv[name_col_tv].map(normalize_name)

    # Dictionnaires matricule → nom
    ifs_dict = dict(zip(df_ifs[id_col_ifs].astype(str), df_ifs["nom_prenom"]))
    tv_dict = dict(zip(df_tv[id_col_tv].astype(str), df_tv["nom_prenom"]))

    # Intersection
    common_ids = set(ifs_dict.keys()) & set(tv_dict.keys())
    print(f"[DEBUG] {len(common_ids)} matricules communs trouvés")

    results = []
    for idx, matricule in enumerate(common_ids, 1):
        nom_ifs = ifs_dict[matricule]
        nom_tv = tv_dict[matricule]
        score = fuzz.ratio(nom_ifs, nom_tv)

        if min_score <= score <= max_score:
            results.append({
                "matricule": matricule,
                "nom_ifs": nom_ifs,
                "nom_topview": nom_tv,
                "score": round(score, 2)
            })

        if idx % 500 == 0:
            print(f"[DEBUG] {idx} comparaisons traitées...")

    print(f"[DEBUG] Comparaisons terminées — {len(results)} correspondances douteuses trouvées")
    print(f"[DEBUG] Temps total : {round(time.time() - start_time, 2)} secondes")

    return {"douteux": results}
