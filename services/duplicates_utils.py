import unidecode
import re
import pandas as pd

COLUMN_MAPPING = {
    "matricule": [
        "matricule", "id_matricule_num", "id_empl", "id_employe",
        "id (matricule num)", "matricule_num", "id", "id_matricule_num"
    ],
    "nom_prenom": [
        "nom_prenom", "nom__prenom", "nom_&_prenom", "nomprenom",
        "nom", "prÃ©nom", "prenom", "prenom_nom", "nom_de_famille",
        "nom utilisateur", "full_name", "displayname"
    ],
    "date_naissance": [
        "date_naissance", "date_naiss", "date_de_naissance",
        "birthdate", "dob", "date_naiss.", "date de naissance"
    ]
}

def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalisation de base
    df.columns = [normalize_column(c) for c in df.columns]

    # Supprimer doublons de colonnes éventuels (ne garder que la 1ère)
    df = df.loc[:, ~df.columns.duplicated()]

    # Harmonisation via le mapping
    for target, variants in COLUMN_MAPPING.items():
        for variant in variants:
            if variant in df.columns:
                print(f"Renommage détecté: '{variant}' -> '{target}'")
                df = df.rename(columns={variant: target})
                break  # Sortir de la boucle une fois qu'on a trouvé une correspondance

    # 🚨 Fusion prénom + nom si pas de colonne "nom_prenom"
    if "nom_prenom" not in df.columns:
        prenom_candidates = [c for c in df.columns if "prenom" in c]
        nom_candidates = [c for c in df.columns if "nom" in c and "prenom" not in c]

        if prenom_candidates and nom_candidates:
            prenom_col = prenom_candidates[0]
            nom_col = nom_candidates[0]
            df["nom_prenom"] = (
                df[prenom_col].astype(str).str.strip()
                + " "
                + df[nom_col].astype(str).str.strip()
            )

    # 🔒 Sécuriser nom_prenom
    if "nom_prenom" in df.columns:
        # Si c'est un DataFrame, extraire la première colonne
        if isinstance(df["nom_prenom"], pd.DataFrame):
            nom_prenom_values = df["nom_prenom"].iloc[:, 0]
        else:
            nom_prenom_values = df["nom_prenom"]
        
        # Reassigner proprement la colonne
        df = df.drop(columns=["nom_prenom"], errors='ignore')
        df["nom_prenom"] = nom_prenom_values.astype(str).str.strip()

    return df


def normalize_column(col_name: str) -> str:
    """
    Nettoie un nom de colonne : minuscule, sans accents, remplace les caractères spéciaux par '_'.
    """
    col = unidecode.unidecode(str(col_name))  # sécurité si col_name n'est pas une string
    col = col.lower()
    col = re.sub(r'[^a-z0-9]+', '_', col)
    col = col.strip("_")
    return col


