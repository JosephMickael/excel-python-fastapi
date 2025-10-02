from fastapi import HTTPException
import pandas as pd
import numpy as np
import math, re, unicodedata
import unidecode
import csv

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
    "matricule": ["matricule", "id (matricule num)", "id empl.", "id employé", "id employe", '"id empl."', "id matricule num", "id_ matricule_num", "id_empl"],
    "sexe": ["sexe", "genre", "sex"],
    "id_personne": ["id personne", "person id"],
    "nom_prenom": ["nom & prenom", "nom utilisateur", "nom affichage interne", "nom affichage externe", "nom_ _prenom", "Travailleur"],
    "nom_de_famille": ["nom de famille", "nom"],
    "prénom": ["prénom", "prenom", "deuxieme prenom"],  # garde séparé
    "date_naissance": ["date naissance", "date de naissance", "date naiss.", "naissance", "date_de_naissance", "date_naissance", "date_naiss"]
}


"""
    Lit un fichier Excel ou CSV et retourne un DataFrame Pandas.
"""
def read_file(file):
    filename = file.filename.lower()

    if filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file.file, dtype=str)
    elif filename.endswith(".csv"):
        # Lecture brute du contenu
        content = file.file.read().decode("utf-8", errors="ignore")
        file.file.seek(0)  # remettre le pointeur au début pour pandas

        # Détection du séparateur
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(content.splitlines()[0])
            sep = dialect.delimiter
        except Exception:
            sep = ","  # fallback par défaut

        df = pd.read_csv(file.file, dtype=str, sep=sep, engine="python")
    else:
        raise ValueError("Format de fichier non supporté")

    # Nettoyer les noms de colonnes
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(r"\s+", "_", regex=True)  # remplace espaces par "_"

    return df
    
def _clean_col(c: str) -> str:
    c = (c or "").strip().replace("\ufeff", "")  # enlève BOM
    c = unicodedata.normalize("NFKD", c).encode("ASCII", "ignore").decode("utf-8")
    c = c.lower()
    # supprime ou remplace ponctuation
    c = re.sub(r"[^\w\s]", " ", c)  # remplace .,"- par espace
    # supprime espaces multiples
    c = re.sub(r"\s+", " ", c).strip()
    return c

def parse_date(val):
    """
    Parse une date en gérant plusieurs formats (%Y-%m-%d, %Y-%m-%d %H:%M:%S, %d/%m/%Y).
    Retourne NaT si impossible.
    """
    if pd.isna(val) or str(val).strip() == "":
        return None

    s = str(val).strip()

    # Détection rapide ISO (commence par 4 chiffres -> année)
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):
        try:
            return pd.to_datetime(s, errors="raise", dayfirst=False)
        except Exception:
            pass

    # Format français ou ambigu -> dayfirst=True
    try:
        return pd.to_datetime(s, errors="raise", dayfirst=True)
    except Exception:
        pass

    # Dernière chance : parsing libre (tolérant)
    return pd.to_datetime(s, errors="coerce")


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


def ensure_key(df, key: str, label: str):
    if key not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Colonne clé '{key}' introuvable dans {label}. Colonnes dispo: {list(df.columns)}"
        )

def clean_column_name(col: str) -> str:
    # Supprimer BOM (\ufeff), espaces, accents et mettre en minuscule ()
    col = col.strip().lower().replace("\ufeff", "")
    col = unicodedata.normalize("NFKD", col).encode("ASCII", "ignore").decode("utf-8")
    return col

# Fonction pour harmoniser les colonnes
def normalize_col(col: str) -> str:
    """ Supprime accents, met en minuscule, nettoie espaces/quotes """
    return unidecode.unidecode(col).replace('"', '').strip().lower()

def harmonize_columns(df):
    """
    Harmonise les colonnes d'un DataFrame selon le MAPPING.
    Renvoie un DataFrame avec colonnes renommées en canonique.
    """
    # 1) clean toutes les colonnes
    df = df.rename(columns={c: _clean_col(c) for c in df.columns})

    # 2) construit le mapping harmonisé
    rename_map = {}
    for canonical, variants in MAPPING.items():
        variants_clean = {_clean_col(v) for v in variants}
        for col in list(df.columns):
            if _clean_col(col) in variants_clean:
                rename_map[col] = canonical

    # 3) applique le mapping
    df = df.rename(columns=rename_map)

    # 4) supprime doublons éventuels
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

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
    if not val_norm:
        return ""

    v = val_norm.strip().lower()

    # Variantes reconnues
    equivalences_f = {"f", "feminin", "féminin", "female", "femme"}
    equivalences_m = {"m", "masculin", "male", "homme"}

    if v in equivalences_f:
        return "F"
    if v in equivalences_m:
        return "M"

    # Si non reconnu, renvoyer en majuscule
    return v.upper()

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
        # dt = pd.to_datetime(val, errors="coerce")
        dt = parse_date(val)
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

# Debut new compare_files

def get_first_value(df, key, col):
    val = df.loc[key, col]
    if isinstance(val, pd.Series):
        return val.iloc[0]
    return val

def compare_files(df1, df2, df3=None, key="matricule"):
    """
    Version optimisée pour de gros fichiers - utilise des opérations vectorisées pandas
    """
    print("DEBUG: début compare_files")
    
    # Harmonisation nom_prenom si pas déjà présent
    for df in [df1, df2, df3]:
        if df is not None and 'nom_prenom' in df.columns:
            df['nom_prenom'] = df['nom_prenom'].astype(str).str.strip().str.lower()

    if df3 is None:
        return compare_two_files_optimized(df1, df2, key)
    
    return compare_three_files_optimized(df1, df2, df3, key)


def compare_two_files_optimized(df1, df2, key="matricule"):
    """Version optimisée pour 2 fichiers"""
    print("DEBUG: comparaison 2 fichiers optimisée")
    
    # Nettoyer les clés (supprimer NaN)
    df1_clean = df1.dropna(subset=[key]).copy()
    df2_clean = df2.dropna(subset=[key]).copy()
    
    common_cols = df1_clean.columns.intersection(df2_clean.columns)
    report_dict = {}

    # 1. MISSING ANALYSIS - Opérations vectorisées
    keys1 = set(df1_clean[key])
    keys2 = set(df2_clean[key])
    
    missing_in_df2_keys = keys1 - keys2
    missing_in_df1_keys = keys2 - keys1
    
    if missing_in_df2_keys:
        missing_df2 = df1_clean[df1_clean[key].isin(missing_in_df2_keys)]
        report_dict['missing_in_df2'] = [
            {col: safe_value(val) for col, val in row.items()}
            for row in missing_df2.to_dict(orient="records")
        ]
    
    if missing_in_df1_keys:
        missing_df1 = df2_clean[df2_clean[key].isin(missing_in_df1_keys)]
        report_dict['missing_in_df1'] = [
            {col: safe_value(val) for col, val in row.items()}
            for row in missing_df1.to_dict(orient="records")
        ]

    # 2. SAME NAME DIFF MATRICULE - Optimisé
    if 'nom_prenom' in df1_clean.columns and 'nom_prenom' in df2_clean.columns:
        same_name_conflicts = find_same_name_conflicts_optimized(df1_clean, df2_clean, key)
        if same_name_conflicts:
            report_dict['same_name_diff_matricule'] = same_name_conflicts

    # 3. DIFF LINE BY LINE - Optimisé avec merge
    common_keys = keys1.intersection(keys2)
    if common_keys:
        diff_results = find_differences_optimized(df1_clean, df2_clean, common_keys, common_cols, key)
        if diff_results:
            report_dict['diff_line_by_line'] = diff_results

    print("DEBUG: comparaison 2 fichiers terminée")
    return clean_report(report_dict)


def compare_three_files_optimized(df1, df2, df3, key="matricule"):
    """Version optimisée pour 3 fichiers"""
    print("DEBUG: comparaison 3 fichiers optimisée")
    
    # Nettoyer les clés
    df1_clean = df1.dropna(subset=[key]).copy()
    df2_clean = df2.dropna(subset=[key]).copy() 
    df3_clean = df3.dropna(subset=[key]).copy()
    
    report_dict = {}
    
    # Sets de clés pour opérations rapides
    keys1 = set(df1_clean[key])
    keys2 = set(df2_clean[key]) 
    keys3 = set(df3_clean[key])
    all_keys = keys1.union(keys2).union(keys3)
    
    print(f"DEBUG: {len(keys1)} clés df1, {len(keys2)} clés df2, {len(keys3)} clés df3")

    # 1. MISSING ANALYSIS - Vectorisé
    missing_in_df1_keys = (keys2.union(keys3)) - keys1
    missing_in_df2_keys = (keys1.union(keys3)) - keys2
    missing_in_df3_keys = (keys1.union(keys2)) - keys3
    
    if missing_in_df1_keys:
        # Prendre les données de df2 ou df3
        source_df = df2_clean if keys2.intersection(missing_in_df1_keys) else df3_clean
        missing_df1 = source_df[source_df[key].isin(missing_in_df1_keys)]
        report_dict['missing_in_df1'] = [
            {col: safe_value(val) for col, val in row.items()}
            for row in missing_df1.to_dict(orient="records")
        ]
    
    if missing_in_df2_keys:
        source_df = df1_clean if keys1.intersection(missing_in_df2_keys) else df3_clean
        missing_df2 = source_df[source_df[key].isin(missing_in_df2_keys)]
        report_dict['missing_in_df2'] = [
            {col: safe_value(val) for col, val in row.items()}
            for row in missing_df2.to_dict(orient="records")
        ]
    
    if missing_in_df3_keys:
        source_df = df1_clean if keys1.intersection(missing_in_df3_keys) else df2_clean
        missing_df3 = source_df[source_df[key].isin(missing_in_df3_keys)]
        report_dict['missing_in_df3'] = [
            {col: safe_value(val) for col, val in row.items()}
            for row in missing_df3.to_dict(orient="records")
        ]
    
    print("DEBUG: missing analysis terminée")

    # 2. SAME NAME DIFF MATRICULE - Optimisé pour 3 fichiers
    if all('nom_prenom' in df.columns for df in [df1_clean, df2_clean, df3_clean]):
        same_name_conflicts = find_same_name_conflicts_3files_optimized(df1_clean, df2_clean, df3_clean, key)
        if same_name_conflicts:
            report_dict['same_name_diff_matricule'] = same_name_conflicts
    
    print("DEBUG: same name conflicts terminé")

    # 3. DIFF LINE BY LINE - Optimisé avec triple merge
    common_cols = df1_clean.columns.intersection(df2_clean.columns).intersection(df3_clean.columns)
    keys_in_all_3 = keys1.intersection(keys2).intersection(keys3)
    
    if keys_in_all_3 and len(common_cols) > 1:  # > 1 car on a au minimum la clé
        diff_results = find_differences_3files_optimized(df1_clean, df2_clean, df3_clean, keys_in_all_3, common_cols, key)
        if diff_results:
            report_dict['diff_line_by_line'] = diff_results
    
    print("DEBUG: diff line by line terminé")
    print("DEBUG: prêt à retourner avec structure originale")
    return clean_report(report_dict)


def find_same_name_conflicts_optimized(df1, df2, key):
    """Version optimisée pour détecter les conflits de noms"""
    # Grouper par nom une seule fois
    df1_grouped = df1.groupby('nom_prenom').agg({
        key: lambda x: list(x.dropna().unique()),
        'date_naissance': lambda x: next((normalize_date(d) for d in x if normalize_date(d)), None)
    }).reset_index()
    
    df2_grouped = df2.groupby('nom_prenom').agg({
        key: lambda x: list(x.dropna().unique()),  
        'date_naissance': lambda x: next((normalize_date(d) for d in x if normalize_date(d)), None)
    }).reset_index()
    
    # Merge sur les noms communs
    merged = df1_grouped.merge(df2_grouped, on='nom_prenom', suffixes=('_df1', '_df2'))
    
    conflicts = []
    for _, row in merged.iterrows():
        mat1_set = set(row[f'{key}_df1'])
        mat2_set = set(row[f'{key}_df2'])
        
        # Conflit si matricules différents mais même date
        if (mat1_set != mat2_set and 
            row['date_naissance_df1'] == row['date_naissance_df2'] and
            not (mat1_set <= mat2_set or mat2_set <= mat1_set)):
            
            conflicts.append({
                "nom_prenom": safe_value(row['nom_prenom']),
                "df1": [{"matricule": safe_value(m), "date_naissance": safe_value(row['date_naissance_df1'])} 
                       for m in row[f'{key}_df1']],
                "df2": [{"matricule": safe_value(m), "date_naissance": safe_value(row['date_naissance_df2'])} 
                       for m in row[f'{key}_df2']]
            })
    
    return conflicts

def normalize_nom_prenom(value: str) -> str:
    """Transforme un nom/prénom en forme normalisée pour comparaison (tokens triés)."""
    if not value:
        return ""
    tokens = tokens_name(value)  # ta fonction existante
    return " ".join(sorted(tokens))  # ordre stable

def find_same_name_conflicts_3files_optimized(df1, df2, df3, key):
    """Version optimisée pour 3 fichiers avec normalisation des noms."""
    conflicts = []

    # Collecter tous les noms uniques (déjà normalisés)
    all_names = set()
    all_names.update(df1['nom_prenom'].dropna().apply(normalize_nom_prenom).unique())
    all_names.update(df2['nom_prenom'].dropna().apply(normalize_nom_prenom).unique())
    all_names.update(df3['nom_prenom'].dropna().apply(normalize_nom_prenom).unique())

    # Traitement par chunks pour éviter la mémoire
    chunk_size = 1000
    name_chunks = [list(all_names)[i:i + chunk_size] for i in range(0, len(all_names), chunk_size)]

    for chunk in name_chunks:
        # Normaliser les noms dans chaque df
        df1_chunk = df1.copy()
        df1_chunk["nom_norm"] = df1_chunk["nom_prenom"].apply(normalize_nom_prenom)

        df2_chunk = df2.copy()
        df2_chunk["nom_norm"] = df2_chunk["nom_prenom"].apply(normalize_nom_prenom)

        df3_chunk = df3.copy()
        df3_chunk["nom_norm"] = df3_chunk["nom_prenom"].apply(normalize_nom_prenom)

        for name in chunk:
            files_data = {}

            # Collecter rapidement pour chaque fichier
            for i, df_chunk in enumerate([df1_chunk, df2_chunk, df3_chunk], 1):
                rows = df_chunk[df_chunk["nom_norm"] == name]
                if not rows.empty:
                    matricules = rows[key].dropna().unique()
                    date = next(
                        (normalize_date(d) for d in rows["date_naissance"].dropna()
                         if normalize_date(d)), None
                    )

                    if len(matricules) > 0:
                        files_data[f'df{i}'] = [
                            {"matricule": safe_value(m), "date_naissance": safe_value(date)}
                            for m in matricules
                        ]

            # Toujours inclure df1, df2, df3 même vides
            for i in range(1, 4):
                if f'df{i}' not in files_data:
                    files_data[f'df{i}'] = []

            # Analyser les conflits si présent dans au moins 2 fichiers
            non_empty = {k: v for k, v in files_data.items() if v}
            if len(non_empty) >= 2:
                matricule_sets = {file_name: {rec["matricule"] for rec in records}
                                  for file_name, records in non_empty.items()}

                unique_sets = list(set(frozenset(s) for s in matricule_sets.values()))

                # Dates cohérentes ?
                dates = [records[0]["date_naissance"] for records in non_empty.values()
                         if records and records[0]["date_naissance"]]
                unique_dates = list(set(dates))

                if len(unique_sets) > 1 and len(unique_dates) <= 1:
                    conflict_entry = {"nom_prenom": safe_value(name)}
                    conflict_entry.update(files_data)
                    conflicts.append(conflict_entry)

    return conflicts


def find_differences_optimized(df1, df2, common_keys, common_cols, key):
    """Version optimisée pour trouver les différences avec merge"""
    # Filtrer sur les clés communes
    df1_common = df1[df1[key].isin(common_keys)].set_index(key)
    df2_common = df2[df2[key].isin(common_keys)].set_index(key)
    
    differences = []
    
    # Comparer colonne par colonne (plus efficace que ligne par ligne)
    for col in common_cols:
        if col == key:  # Skip la clé elle-même
            continue
            
        if col in df1_common.columns and col in df2_common.columns:
            # Alignment automatique par index
            series1 = df1_common[col]
            series2 = df2_common[col]
            
            # Indexes communs
            common_idx = series1.index.intersection(series2.index)
            
            for idx in common_idx:
                # val1 = safe_value(series1.loc[idx])
                # val2 = safe_value(series2.loc[idx]) 
                val1 = safe_value(series1.loc[idx].iloc[0] if isinstance(series1.loc[idx], pd.Series) else series1.loc[idx])
                val2 = safe_value(series2.loc[idx].iloc[0] if isinstance(series2.loc[idx], pd.Series) else series2.loc[idx])

                
                if not values_equal_smart(col, val1, val2):
                    differences.append({
                        "key": safe_value(idx),
                        "column": col,
                        "df1_value": val1,
                        "df2_value": val2,
                    })
    
    return differences


def extract_scalar(df, key, col):
    try:
        val = df.loc[key, col]
        if isinstance(val, (pd.Series, list)):
            # garde juste la première valeur si doublon
            return val.iloc[0] if hasattr(val, "iloc") else val[0]
        return val
    except Exception:
        return None

def find_differences_3files_optimized(df1, df2, df3, common_keys, common_cols, key):
    """Compare 3 fichiers et retourne uniquement les vraies différences (avec filtrage intelligent)."""

    # Indexation sur la clé
    df1_indexed = df1[df1[key].isin(common_keys)].set_index(key)
    df2_indexed = df2[df2[key].isin(common_keys)].set_index(key)
    df3_indexed = df3[df3[key].isin(common_keys)].set_index(key)

    differences = []

    key_list = list(common_keys)
    chunk_size = 5000

    # helper pour récupérer une valeur scalaire
    def extract_scalar(df, k, col):
        try:
            val = df.loc[k, col]
            if isinstance(val, (pd.Series, list)):
                return val.iloc[0] if hasattr(val, "iloc") else val[0]
            return val
        except Exception:
            return None

    for i in range(0, len(key_list), chunk_size):
        chunk_keys = key_list[i:i + chunk_size]

        for col in common_cols:
            if col == key:
                continue

            if all(col in df.columns for df in [df1_indexed, df2_indexed, df3_indexed]):
                for k in chunk_keys:
                    if k in df1_indexed.index and k in df2_indexed.index and k in df3_indexed.index:
                        val1 = safe_value(extract_scalar(df1_indexed, k, col))
                        val2 = safe_value(extract_scalar(df2_indexed, k, col))
                        val3 = safe_value(extract_scalar(df3_indexed, k, col))

                        # si toutes les valeurs sont équivalentes -> on ignore
                        if values_equal_smart(col, val1, val2) and values_equal_smart(col, val1, val3):
                            continue

                        # si au moins 2 valeurs sont différentes, on ajoute
                        normalized_set = set(
                            v for v in [norm_basic_for_cmp(val1), norm_basic_for_cmp(val2), norm_basic_for_cmp(val3)]
                            if v not in (None, "", "nan")
                        )
                        if len(normalized_set) > 1:
                            differences.append({
                                "key": safe_value(k),
                                "column": col,
                                "df1_value": val1,
                                "df2_value": val2,
                                "df3_value": val3
                            })

    return differences

# Fin new compare_files



# MINE VISITS
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
    
def read_topview_file(file):
    filename = file.filename.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(
            file.file,
            sep=None,           
            engine="python",    
            on_bad_lines="skip",   
            encoding="utf-8",
            dtype=str
        )
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        # au cas où un TopView sortirait en Excel
        return pd.read_excel(file.file, dtype=str)
    else:
        raise ValueError("Format de fichier TopView non supporté")
    
    
def normalize_name(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().upper()
    s = s.replace(",", " ")  
    s = s.replace("-", " ")  
    s = " ".join(s.split()) 
    # enlever les accents
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s