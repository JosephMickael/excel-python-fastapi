import pandas as pd
import numpy as np
import unicodedata
import re
import math, re, unicodedata
import pandas as pd


# Mapping des colonnes standard (normalisation des colonnes)
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

"""
    Lit un fichier Excel ou CSV et retourne un DataFrame Pandas.
"""
def read_file(file):
    name = file.filename.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file.file)
    else:
        return pd.read_excel(file.file)
    
    
"""
    Transforme les colonnes prénom + nom de famille en nom_prenom
    ou sépare nom_prenom en deux colonnes si nécessaire.
"""
def unify_name_column(df):
    # Si les colonnes séparées existent
    if "prénom" in df.columns and "nom de famille" in df.columns:
        df["nom_prenom"] = df["nom de famille"].astype(str) + ", " + df["prénom"].astype(str)
    # Si la colonne combinée existe et que tu veux éventuellement séparer
    elif "nom_prenom" in df.columns:
        # Sépare par la virgule si nécessaire
        split_names = df["nom_prenom"].str.split(",", n=1, expand=True)
        if split_names.shape[1] == 2:
            df["nom_de_famille"] = split_names[0].str.strip()
            df["prénom"] = split_names[1].str.strip()
    return df


# Fonction pour harmoniser les colonnes
def harmonize_columns(df, mapping=MAPPING):
    df.columns = [col.strip().lower() for col in df.columns]
    new_cols = {}
    for standard_name, variants in mapping.items():
        for v in variants:
            if v.lower() in df.columns:
                new_cols[v.lower()] = standard_name
                break
    df = df.rename(columns=new_cols)
    return df

def safe_value(val):
    """Convertit toutes les valeurs en types JSON-compatibles"""
    if pd.isna(val):
        return None
    if isinstance(val, (np.integer, np.int64)):
        return int(val)
    if isinstance(val, (np.floating, np.float64)):
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    return val

def harmonize_name(df, first_name_col=None, last_name_col=None, full_name_col=None):
    """
    Crée une colonne 'nom_prenom' harmonisée pour comparer les fichiers
    """
    if full_name_col:
        # Convertit "Nom, Prénom" ou "Nom Prénom" en lowercase strip
        df['nom_prenom'] = df[full_name_col].str.strip().str.lower()
    elif first_name_col and last_name_col:
        df['nom_prenom'] = (df[first_name_col].str.strip() + ", " + df[last_name_col].str.strip()).str.lower()
    else:
        raise ValueError("Il faut spécifier soit full_name_col soit first_name_col et last_name_col")
    return df


def to_str(x):
    """Convertit proprement en str pour éviter les None / NaN."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x)

def strip_accents(s: str) -> str:
    """Supprime les accents pour comparaison insensible."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def norm_basic_for_cmp(x) -> str:
    """
    Normalisation générique pour comparaison :
    - cast en str
    - trim
    - supprime accents
    - insensible casse
    - espaces multiples -> un espace
    - retire ponctuation légère
    """
    s = to_str(x).strip()
    if not s:
        return ""
    s = strip_accents(s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.,;:/\\]+", " ", s)
    s = re.sub(r"[\-_]", "", s)
    return s.casefold().strip()

def tokens_name(x) -> set:
    """Découpe un nom en tokens normalisés."""
    s = norm_basic_for_cmp(x)
    return set(t for t in s.split(" ") if t)

def map_sexe(val_norm: str) -> str:
    """Uniformise les valeurs de sexe."""
    if val_norm in {"f", "feminin", "féminin"}:
        return "F"
    if val_norm in {"m", "masculin"}:
        return "M"
    return val_norm.upper()

def values_equal_smart(col: str, v1, v2) -> bool:
    """Comparaison intelligente par colonne."""
    if to_str(v1) == "" and to_str(v2) == "":
        return True

    # Numériques
    try:
        f1 = float(to_str(v1).replace(",", "."))
        f2 = float(to_str(v2).replace(",", "."))
        if math.isfinite(f1) and math.isfinite(f2):
            return abs(f1 - f2) < 1e-12
    except Exception:
        pass

    col_norm = strip_accents(col).casefold().strip()

    # Sexe
    if col_norm in {"sexe", "genre", "sex"}:
        return map_sexe(norm_basic_for_cmp(v1)) == map_sexe(norm_basic_for_cmp(v2))

    # Identifiants
    if any(k in col_norm for k in ["id_personne", "idposte", "id poste", "id_emploi", "id emploi"]):
        a = re.sub(r"\s+", "", norm_basic_for_cmp(v1))
        b = re.sub(r"\s+", "", norm_basic_for_cmp(v2))
        return a == b

    # Noms / prénoms
    if col_norm in {"nom_prenom", "nom prenom", "nom", "prenom", "prénom", "nom_de_famille"}:
        t1, t2 = tokens_name(v1), tokens_name(v2)
        if not t1 or not t2:
            return norm_basic_for_cmp(v1) == norm_basic_for_cmp(v2)
        inter = len(t1 & t2)
        union = len(t1 | t2)
        if inter > 0 and (t1.issubset(t2) or t2.issubset(t1) or inter / union >= 0.6):
            return True
        return False

    # Par défaut
    return norm_basic_for_cmp(v1) == norm_basic_for_cmp(v2)

def compare_files(df1, df2, key="matricule"):
    report = []

    # Harmonisation nom_prenom si pas déjà présent
    if 'nom_prenom' not in df1.columns:
        df1['nom_prenom'] = df1['nom_prenom'].str.strip().str.lower()
    if 'nom_prenom' not in df2.columns:
        df2['nom_prenom'] = df2['nom_prenom'].str.strip().str.lower()

    common_cols = df1.columns.intersection(df2.columns)

    # Lignes manquantes
    missing_in_df2 = df1[~df1[key].isin(df2[key])]
    missing_in_df1 = df2[~df2[key].isin(df1[key])]

    report_dict = {}

    if not missing_in_df2.empty:
        report_dict['missing_in_df2'] = [
            {col: safe_value(val) for col, val in row.items()}
            for row in missing_in_df2.to_dict(orient="records")
        ]

    if not missing_in_df1.empty:
        report_dict['missing_in_df1'] = [
            {col: safe_value(val) for col, val in row.items()}
            for row in missing_in_df1.to_dict(orient="records")
        ]

    # Convertit un set en valeur unique si un seul élément, sinon en liste
    def to_scalar_or_list(values):
        values = list(values)
        values = [str(v).strip() for v in values]  # 🔑 conversion en texte
        if len(values) == 1:
            return values[0]
        return values

    # Même nom/prénom mais matricules différents
    same_name_diff_matricule = []

    df1_grouped = df1.groupby("nom_prenom")[key].unique()
    df2_grouped = df2.groupby("nom_prenom")[key].unique()

    for name in set(df1_grouped.index).intersection(df2_grouped.index):
        mat1_list = set([str(v).strip() for v in df1_grouped[name]])
        mat2_list = set([str(v).strip() for v in df2_grouped[name]])
        if mat1_list != mat2_list:  # maintenant vraie différence uniquement
            same_name_diff_matricule.append({
                "nom_prenom": safe_value(name),
                f"{key}_df1": to_scalar_or_list(mat1_list),
                f"{key}_df2": to_scalar_or_list(mat2_list)
            })

    if same_name_diff_matricule:
        report_dict['same_name_diff_matricule'] = same_name_diff_matricule



    # Différences colonne par colonne pour les clés communes
    # common_keys = df1[key].isin(df2[key])
    common_keys = df1[key].isin(df2[key]) & df1[key].notna()
    diff_line_by_line = []

    for k in df1[key][common_keys]:
        rows1 = df1[df1[key] == k]
        rows2 = df2[df2[key] == k]

        if rows1.empty or rows2.empty:
            continue 

        row1 = rows1.iloc[0]
        row2 = rows2.iloc[0]
        
        for col in common_cols:
            val1 = safe_value(row1[col]) if col in row1 else None
            val2 = safe_value(row2[col]) if col in row2 else None
            if not values_equal_smart(col, val1, val2):
                diff_line_by_line.append({
                    "key": safe_value(k),
                    "column": col,
                    "df1_value": safe_value(val1),
                    "df2_value": safe_value(val2),
                })
    if diff_line_by_line:
        report_dict['diff_line_by_line'] = diff_line_by_line

    return report_dict

