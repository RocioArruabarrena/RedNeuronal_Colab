"""
Genera friendships sinteticas para profiles_final.csv, con la MISMA logica
que ya usaste en extract_from_yo_adolescente.py: los pares "amigos" (is_mutual=true)
tienen que salir mayormente de usuarios parecidos (mismos intereses / misma
subcultura), y los pares "no amigos" (is_mutual=false) mayormente de usuarios
sin nada en comun. Si esto fuera puro random, el modelo no tendria señal real
para aprender.

Solo usa la libreria estandar (csv, json, random) - sin pandas.
"""
import csv
import json
import random

PROFILES_FILE = "profiles_final.csv"
OUTPUT_FILE = "friendships_final.csv"

# ---- Ajusta estos parametros a gusto ----
N_POSITIVOS = 2000          # cantidad de pares is_mutual=true
N_NEGATIVOS = 2000          # cantidad de pares is_mutual=false
PROP_SIMILARES_EN_POS = 0.8   # % de positivos que deben salir del pool "parecidos"
PROP_DISTINTOS_EN_NEG = 0.85  # % de negativos que deben salir del pool "distintos"
UMBRAL_SIMILITUD = 0.15       # score arriba de esto se considera "parecidos"
SEED = 42
# ------------------------------------------

random.seed(SEED)


def leer_perfiles(path):
    with open(path, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    for fila in filas:
        fila["interests"] = json.loads(fila["interests"])
    return filas


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def score_similitud(perfil_a, perfil_b):
    sim_intereses = jaccard(perfil_a["interests"], perfil_b["interests"])
    mismo_avatar = perfil_a["pixel_avatar"] == perfil_b["pixel_avatar"]
    return sim_intereses + (0.3 if mismo_avatar else 0)


def generar_pares(perfiles, cantidad, quiere_similares, proporcion_objetivo, pares_usados):
    ids = [p["id"] for p in perfiles]
    perfil_por_id = {p["id"]: p for p in perfiles}
    resultado = []
    intentos = 0
    max_intentos = cantidad * 80

    while len(resultado) < cantidad and intentos < max_intentos:
        intentos += 1
        a, b = random.sample(ids, 2)
        clave = tuple(sorted((a, b)))
        if clave in pares_usados:
            continue

        score = score_similitud(perfil_por_id[a], perfil_por_id[b])
        es_similar = score > UMBRAL_SIMILITUD

        # Aceptamos el par si coincide con lo que buscamos (similar/distinto),
        # o con una probabilidad baja igual lo dejamos pasar (ruido real).
        cumple_objetivo = es_similar if quiere_similares else not es_similar
        aceptar = cumple_objetivo or random.random() < (1 - proporcion_objetivo)

        if aceptar:
            pares_usados.add(clave)
            resultado.append((a, b))

    return resultado


def main():
    perfiles = leer_perfiles(PROFILES_FILE)
    pares_usados = set()

    positivos = generar_pares(
        perfiles, N_POSITIVOS, quiere_similares=True,
        proporcion_objetivo=PROP_SIMILARES_EN_POS, pares_usados=pares_usados,
    )
    negativos = generar_pares(
        perfiles, N_NEGATIVOS, quiere_similares=False,
        proporcion_objetivo=PROP_DISTINTOS_EN_NEG, pares_usados=pares_usados,
    )

    filas_salida = []
    for a, b in positivos:
        filas_salida.append({"requester_id": a, "receiver_id": b, "is_mutual": "true"})
    for a, b in negativos:
        filas_salida.append({"requester_id": a, "receiver_id": b, "is_mutual": "false"})

    random.shuffle(filas_salida)
    for i, fila in enumerate(filas_salida, start=1):
        fila["id"] = i

    campos = ["id", "requester_id", "receiver_id", "is_mutual"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas_salida)

    print(
        f"Listo: {OUTPUT_FILE} con {len(filas_salida)} filas "
        f"({len(positivos)} is_mutual=true / {len(negativos)} is_mutual=false)"
    )


if __name__ == "__main__":
    main()
