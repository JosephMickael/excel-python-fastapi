import re
import unicodedata
import pandas as pd

MAPPING = {
    "statut": ["statut", "statut_du_travail", "statut du travail"],
    "matricule": [
        "matricule",
        "id_matricule_num",
        "id__matricule_num",   # 👈 ajouté
        "id (matricule num)",
        "id empl.",
        "id_employe",
        "id_employé",
        "id employe",
        "id employé"
    ],
    "sexe": ["sexe", "genre", "sex"],
    "id_personne": ["id_personne", "person_id", "id personne"],
    "nom_prenom": [
        "nom_prenom",
        "nom_utilisateur",
        "nom & prenom",
        "nom_affichage_interne",
        "nom_affichage_externe"
    ],
    "nom": ["nom", "nom_de_famille", "nom de famille"],
    "prenom": ["prenom", "prénom", "deuxieme_prenom", "deuxième_prenom"],
    "date_naissance": [
        "date_naissance",
        "date_naiss",
        "date_naiss.",
        "date de naissance",
        "naissance"
    ]
}


def _norm(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _get_first_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Retourne la première colonne qui existe parmi candidates, toujours en Series"""
    for c in candidates:
        if c in df.columns:
            col = df[c]
            # Si DataFrame → prendre la première colonne
            if isinstance(col, pd.DataFrame):
                return col.iloc[:, 0]
            return col
    return pd.Series([""] * len(df))

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normaliser noms colonnes
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )

    # Construire mapping
    rename_dict = {}
    for target, variants in MAPPING.items():
        for v in variants + [target]:
            v_clean = _norm(v)
            for c in df.columns:
                if _norm(c) == v_clean:
                    rename_dict[c] = target

    df = df.rename(columns=rename_dict)

    # 🚑 Supprimer colonnes dupliquées après mapping
    df = df.loc[:, ~df.columns.duplicated()]

    # Colonnes minimales
    for col in ["matricule", "nom_prenom", "date_naissance", "nom_de_famille", "prénom"]:
        if col not in df.columns:
            df[col] = ""

    # Vérifier si nom_prenom est vide ou inexistant
    needs_build = (
        "nom_prenom" not in df.columns
        or df["nom_prenom"].fillna("").astype(str).str.strip().eq("").all()
    )

    if needs_build:
        nom = _get_first_series(df, ["nom_de_famille", "nom"]).fillna("").astype(str)
        prenom = _get_first_series(df, ["prénom", "prenom"]).fillna("").astype(str)
        deux = _get_first_series(df, ["deuxieme_prenom"]).fillna("").astype(str)

        full_firstnames = (prenom.str.strip() + " " + deux.str.strip()).str.strip()
        df["nom_prenom"] = (nom.str.strip() + ", " + full_firstnames).str.strip(", ").str.lower()
    else:
        df["nom_prenom"] = df["nom_prenom"].fillna("").astype(str).str.lower().str.strip()

    # Standardiser date
    if "date_naissance" in df.columns:
        dates = pd.to_datetime(df["date_naissance"], errors="coerce", dayfirst=True)
        df["date_naissance"] = dates.dt.strftime("%Y-%m-%d").fillna("")

    # Nettoyer matricule
    if "matricule" in df.columns:
        df["matricule"] = df["matricule"].fillna("").astype(str).str.strip()

    return df


def ensure_nom_prenom(df):
    """
    Construit une vraie colonne nom_prenom si besoin
    """
    if "prenom" in df.columns and "nom" in df.columns:
        df["nom_prenom_clean"] = (
            df["prenom"].fillna("").astype(str).str.strip() + " " +
            df["nom"].fillna("").astype(str).str.strip()
        ).str.strip()
    elif "nom_de_famille" in df.columns and "prénom" in df.columns:
        df["nom_prenom_clean"] = (
            df["prénom"].fillna("").astype(str).str.strip() + " " +
            df["nom_de_famille"].fillna("").astype(str).str.strip()
        ).str.strip()
    else:
        df["nom_prenom_clean"] = df.get("nom_prenom", "")

    return df


def find_duplicates(df): 

    df = ensure_nom_prenom(df)

    keys = []
    if "matricule" in df.columns:
        keys.append("matricule")
    if "id_personne" in df.columns:
        keys.append("id_personne")
    if "nom_prenom_clean" in df.columns:
        keys.append("nom_prenom_clean")

    if not keys:
        return {"count": 0, "rows": []}

    duplicates = df[df.duplicated(subset=keys, keep=False)]

    # Remplacer NaN par None
    duplicates = duplicates.where(pd.notnull(duplicates), None)

    # Regrouper par les clés pour montrer les doublons ensemble
    grouped = []
    for _, group in duplicates.groupby(keys):
        grouped.append(group.to_dict(orient="records"))

    return {
        "count": len(duplicates),
        "groups": grouped
    }
