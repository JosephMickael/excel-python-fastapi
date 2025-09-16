from fastapi import HTTPException
import pandas as pd
import numpy as np
import unicodedata
import re
import math, re, unicodedata
import pandas as pd
from datetime import datetime


# Mapping des colonnes standard (normalisation des colonnes)
# MAPPING = {
#     "statut": ["statut", "Statut du travail"],
#     "matricule": ["matricule", "ID (Matricule Num)", "ID empl.", "ID employé"],
#     "sexe": ["sexe", "Sexe"],
#     "id_personne": ["id personne", "Person ID"],
#     "nom_prenom": ["nom & prenom", "Nom utilisateur", "Nom de famille", "Prénom", "Nom", "Noms", "Prénoms", "Prénom"],
#     "date_naissance": ["date naissance", "Date de naissance"],
#     "date_emploi": ["date emploi", "Date de début"],
#     "date_fin_emploi": ["date fin emploi", "Date de fin"],
#     "departement": ["departement", "Département"],
#     "poste": ["titre poste", "Poste"],
#     "sous_departement": ["sous departement", "Sous-département"],
# }

MAPPING = {
    "statut": ["statut", "statut du travail"],
    "matricule": ["matricule", "id (matricule num)", "id empl.", "id employé", "id employe", '"id empl."'],
    "sexe": ["sexe", "genre", "sex"],
    "id_personne": ["id personne", "person id"],
    "nom_prenom": ["nom & prenom", "nom utilisateur", "nom affichage interne", "nom affichage externe"],
    "nom_de_famille": ["nom de famille", "nom"],
    "prénom": ["prénom", "prenom", "deuxieme prenom"],  # garde séparé
    "date_naissance": ["date naissance", "date de naissance", "date naiss.", "naissance"]
}


"""
    Lit un fichier Excel ou CSV et retourne un DataFrame Pandas.
"""
def read_file(file):
    filename = file.filename.lower()
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(file.file, dtype=str)
    elif filename.endswith(".csv"):
        # auto-détection du séparateur
         return pd.read_csv(file.file, dtype=str, sep=None, engine="python")
    else:
        raise ValueError("Format de fichier non supporté")

    # def read_file(file):
    #     name = file.filename.lower()
    #     if name.endswith(".csv"):
    #         return pd.read_csv(file.file)
    #     else:
    #         return pd.read_excel(file.file, dtype=str)
    
    

def _clean_col(c: str) -> str:
    c = (c or "").strip().replace("\ufeff", "")  # enlève BOM
    c = unicodedata.normalize("NFKD", c).encode("ASCII", "ignore").decode("utf-8")
    return c.lower()    
    
"""
    Transforme les colonnes prénom + nom de famille en nom_prenom
    ou sépare nom_prenom en deux colonnes si nécessaire.
"""
def unify_name_column(df):
    cols = set(df.columns)

    if "nom_prenom" in cols:
        s = df["nom_prenom"]
        # si doublon “nom_prenom” (ça peut arriver avant dédoublonnage), garder la première série
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        df["nom_prenom"] = s.astype(str).str.strip().str.lower()

    else:
        left = None
        if "nom_de_famille" in cols:
            left = df["nom_de_famille"].astype(str).str.strip().str.lower()
        elif "nom" in cols:
            left = df["nom"].astype(str).str.strip().str.lower()

        right = None
        if "prénom" in cols:
            right = df["prénom"].astype(str).str.strip().str.lower()
        elif "prenom" in cols:
            right = df["prenom"].astype(str).str.strip().str.lower()

        if left is not None and right is not None:
            df["nom_prenom"] = left + ", " + right
        elif left is not None:
            df["nom_prenom"] = left
        else:
            df["nom_prenom"] = ""  # évite les crashs

    return df

# def unify_name_column(df):
#     # Si les colonnes séparées existent
#     if "prénom" in df.columns and "nom de famille" in df.columns:
#         df["nom_prenom"] = df["nom de famille"].astype(str) + ", " + df["prénom"].astype(str)
#     # Si la colonne combinée existe et que tu veux éventuellement séparer
#     elif "nom_prenom" in df.columns:
#         # Sépare par la virgule si nécessaire
#         split_names = df["nom_prenom"].str.split(",", n=1, expand=True)
#         if split_names.shape[1] == 2:
#             df["nom_de_famille"] = split_names[0].str.strip()
#             df["prénom"] = split_names[1].str.strip()
#     return df

def ensure_key(df, key: str, label: str):
    if key not in df.columns:
        raise HTTPException(status_code=400, detail=f"Colonne clé '{key}' introuvable dans {label}")


def clean_column_name(col: str) -> str:
    # Supprimer BOM (\ufeff), espaces, accents et mettre en minuscule ()
    col = col.strip().lower().replace("\ufeff", "")
    col = unicodedata.normalize("NFKD", col).encode("ASCII", "ignore").decode("utf-8")
    return col

# Fonction pour harmoniser les colonnes
def harmonize_columns(df):
    # 1) nettoie tous les noms
    df = df.rename(columns={c: _clean_col(c) for c in df.columns})

    # 2) construit la table de renommage insensible à la casse/accents
    rename_map = {}
    for canonical, variants in MAPPING.items():
        variants_clean = {_clean_col(v) for v in variants}
        for col in list(df.columns):
            if _clean_col(col) in variants_clean:
                rename_map[col] = canonical

    df = df.rename(columns=rename_map)

    # 3) si des colonnes sont dupliquées après renommage → garde la 1re
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    return df

# def harmonize_columns(df, mapping=MAPPING):
#     df.columns = [col.strip().lower() for col in df.columns]
#     new_cols = {}
#     for standard_name, variants in mapping.items():
#         for v in variants:
#             if v.lower() in df.columns:
#                 new_cols[v.lower()] = standard_name
#                 break
#     df = df.rename(columns=new_cols)
#     return df

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
    
    # Dates (ex: 1999-01-27T00:00:00 == 1999-01-27)
    if col_norm in {"date_naissance", "date de naissance"}:
        d1, d2 = normalize_date(v1), normalize_date(v2)
        return d1 == d2

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

def normalize_date(val):
    """Convertit une valeur en date (format AAAA-MM-JJ)."""
    if val is None or str(val).strip() == "":
        return None
    try:
        dt = pd.to_datetime(val, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.date()  # AAAA-MM-JJ
    except Exception:
        return None

def compare_two_dataframes(dfA, dfB, key="matricule"):
    """Compare deux DataFrames avec la même logique que df1 vs df2"""
    result = {}
    common_cols = dfA.columns.intersection(dfB.columns)

    # Lignes manquantes
    missing_in_B = dfA[~dfA[key].isin(dfB[key])]
    if not missing_in_B.empty:
        result["missing_in_B"] = missing_in_B.to_dict(orient="records")

    missing_in_A = dfB[~dfB[key].isin(dfA[key])]
    if not missing_in_A.empty:
        result["missing_in_A"] = missing_in_A.to_dict(orient="records")

    # Différences ligne par ligne
    diff_line_by_line = []
    common_keys = dfA[key].isin(dfB[key]) & dfA[key].notna()
    for k in dfA[key][common_keys]:
        rowsA = dfA[dfA[key] == k]
        rowsB = dfB[dfB[key] == k]
        if rowsA.empty or rowsB.empty:
            continue
        rowA, rowB = rowsA.iloc[0], rowsB.iloc[0]
        for col in common_cols:
            v1, v2 = safe_value(rowA[col]), safe_value(rowB[col])
            if not values_equal_smart(col, v1, v2):
                diff_line_by_line.append({
                    "key": safe_value(k),
                    "column": col,
                    "dfA_value": v1,
                    "dfB_value": v2
                })
    if diff_line_by_line:
        result["diff_line_by_line"] = diff_line_by_line

    return result

def compare_files(df1, df2, df3=None, key="matricule"):
    report = []
    
    print("DEBUG: début compare_files")

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
    print("DEBUG: missing_in_df1/2 terminés")

    # Même nom/prénom mais matricules différents
    same_name_diff_matricule = []

    df1_grouped = df1.groupby("nom_prenom")
    df2_grouped = df2.groupby("nom_prenom")

    for name in set(df1_grouped.groups.keys()).intersection(df2_grouped.groups.keys()):
        # Récupère toutes les lignes correspondantes dans chaque fichier
        rows1 = df1_grouped.get_group(name)
        rows2 = df2_grouped.get_group(name)

        # Listes (matricule + date_naissance) pour chaque fichier
        df1_records = [
            {
                "matricule": safe_value(row[key]),
                "date_naissance": safe_value(normalize_date(row.get("date_naissance")))
            }
            for _, row in rows1.iterrows()
            if pd.notna(row.get(key))
        ]

        df2_records = [
            {
                "matricule": safe_value(row[key]),
                "date_naissance": safe_value(normalize_date(row.get("date_naissance")))
            }
            for _, row in rows2.iterrows()
            if pd.notna(row.get(key))
        ]

        # Normalisation pour comparaison des sets
        mat1_list = {rec["matricule"] for rec in df1_records}
        mat2_list = {rec["matricule"] for rec in df2_records}

        # Date de naissance (on prend la 1ère non nulle de chaque côté)
        d1 = next((rec["date_naissance"] for rec in df1_records if rec["date_naissance"]), None)
        d2 = next((rec["date_naissance"] for rec in df2_records if rec["date_naissance"]), None)

        # Vérifie si vrai conflit : matricules différents + même date de naissance
        if mat1_list != mat2_list and d1 == d2:
            # Cas à ignorer : si un set est inclus dans l'autre (doublons internes)
            if not (mat1_list <= mat2_list or mat2_list <= mat1_list):
                same_name_diff_matricule.append({
                    "nom_prenom": safe_value(name),
                    "df1": df1_records,
                    "df2": df2_records
                })

    if same_name_diff_matricule:
        report_dict['same_name_diff_matricule'] = same_name_diff_matricule

    print("DEBUG: same_name_diff_matricule terminés")


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
        
    print("DEBUG: diff_line_by_line terminés")
        
    # Si df3 existe
    if df3 is not None:
        report_dict["df1_vs_df3"] = compare_two_dataframes(df1, df3, key)

        report_dict["df2_vs_df3"] = compare_two_dataframes(df2, df3, key)
        
    print("DEBUG: prêt à retourner")
    
    # return report_dict
    return clean_report(report_dict)




def clean_report(obj):
    """Nettoie récursivement le report pour remplacer NaN/NaT/inf par None."""
    if isinstance(obj, dict):
        return {k: clean_report(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_report(x) for x in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    else:
        return obj