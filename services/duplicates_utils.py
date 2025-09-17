import unidecode
import re
import pandas as pd

COLUMN_MAPPING = {
    "matricule": [
        "matricule", "id_matricule_num", "id_empl", "id_employe",
        "id (matricule num)", "matricule_num", "id"
    ],
    "nom_prenom": [
        "nom_prenom", "nom__prenom", "nom_&_prenom", "nomprenom",
        "prenom_nom", "nom utilisateur", "full_name", "displayname"
    ],
    "nom": [
        "nom", "nom_de_famille", "last name", "lastname"
    ],
    "prenom": [
        "prenom", "prénom", "prenoms", "deuxieme prenom", "first name", "firstname"
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
                # print(f"Renommage détecté: '{variant}' -> '{target}'")
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


def to_str(x):
    # Convertit proprement en str en gérant None/NaN
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x)

def _normalize_fullname(s: str) -> str:
    s = to_str(s).strip()
    if not s:
        return ""
    # Cas "Nom, Prénom" → "Prénom Nom"
    m = re.match(r"^\s*([^,]+)\s*,\s*(.+)$", s)
    if m:
        last, first = m.group(1), m.group(2)
        s = f"{first} {last}"
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _norm_token(x):
    # vers str, trim, minuscule, compresse espaces
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip().lower()



def unify_name_column(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)

    # 0) Dé-doublonner les colonnes
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # 🔑 Cas spécial : si à la fois nom et prénom existent → on reconstruit
    has_nom = ("nom" in cols) or ("nom_de_famille" in cols)
    has_prenom = ("prenom" in cols) or ("prénom" in cols)

    if has_nom and has_prenom:
        nom_col = "nom_de_famille" if "nom_de_famille" in cols else "nom"
        prenom_col = "prénom" if "prénom" in cols else "prenom"
        df["nom_prenom"] = (
            df[nom_col].astype(str).str.strip().str.lower()
            + " "
            + df[prenom_col].astype(str).str.strip().str.lower()
        )
        df["nom_prenom"] = df["nom_prenom"].str.replace(r"\s+", " ", regex=True)
        return df

    # Sinon → garder nom_prenom existant si présent
    if "nom_prenom" in cols:
        s = df["nom_prenom"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        df["nom_prenom"] = s.astype(str).str.strip().str.lower()
        return df

    # Fallbacks
    if "nom" in cols:
        df["nom_prenom"] = df["nom"].astype(str).str.strip().str.lower()
    elif "nom_de_famille" in cols:
        df["nom_prenom"] = df["nom_de_famille"].astype(str).str.strip().str.lower()
    elif "prenom" in cols or "prénom" in cols:
        col = "prénom" if "prénom" in cols else "prenom"
        df["nom_prenom"] = df[col].astype(str).str.strip().str.lower()
    else:
        df["nom_prenom"] = ""

    return df
