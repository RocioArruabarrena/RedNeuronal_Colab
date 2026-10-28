# scripts/features.py
"""
Calculo de las 5 features de un par de perfiles, compartido por los
scripts de datos (split_dataset_csv.py, etc.).

Features por par de perfiles:
- similitud_intereses: Jaccard entre tags de perfil + subcultura (interests)
- diff_activity: diferencia absoluta de activity_score
- similitud_tags_posts: Jaccard entre hashtags de posts publicos
  (tags_agregados)
- tiene_tags_posts: 1 si ambos perfiles tienen al menos un tag de posts,
  0 si a alguno le falta (para no confundir "sin datos" con "sin afinidad")
- mismo_avatar_subcultura: 1 si comparten el mismo tag "subcultura:X",
  si no 0

El orden de FEATURE_NAMES es el orden de entrada de la red.
"""

FEATURE_NAMES = [
    "similitud_intereses",
    "diff_activity",
    "similitud_tags_posts",
    "tiene_tags_posts",
    "mismo_avatar_subcultura",
]


def extraer_subcultura(intereses: list[str] | None) -> str | None:
    if not intereses:
        return None
    for tag in intereses:
        if isinstance(tag, str) and tag.startswith("subcultura:"):
            return tag.split(":", 1)[1]
    return None


def jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def calcular_features(perfil_a, perfil_b) -> list[float]:
    """
    perfil_a y perfil_b: cualquier objeto con los atributos interests,
    activity_score y tags_agregados.
    """
    intereses_a = set(perfil_a.interests or [])
    intereses_b = set(perfil_b.interests or [])
    similitud_intereses = jaccard(intereses_a, intereses_b)

    diff_activity = abs((perfil_a.activity_score or 0) - (perfil_b.activity_score or 0))

    tags_posts_a = set(perfil_a.tags_agregados or [])
    tags_posts_b = set(perfil_b.tags_agregados or [])
    similitud_tags_posts = jaccard(tags_posts_a, tags_posts_b)
    tiene_tags_posts = 1.0 if tags_posts_a and tags_posts_b else 0.0

    sub_a = extraer_subcultura(perfil_a.interests)
    sub_b = extraer_subcultura(perfil_b.interests)
    mismo_avatar_subcultura = 1.0 if sub_a and sub_b and sub_a == sub_b else 0.0

    return [
        similitud_intereses,
        diff_activity,
        similitud_tags_posts,
        tiene_tags_posts,
        mismo_avatar_subcultura,
    ]
