"""
Procesa el CSV crudo que baja Mockaroo y lo deja listo para el proyecto:
- interests y tags_agregados pasan de un solo valor a un array JSON (2-4 valores por usuario),
  necesario para que la similitud Jaccard entre usuarios no sea siempre 0 o 1.
- is_online / is_moderator se reasignan con la proporcion real que quieras
  (Mockaroo los generaba mas o menos 50/50, algo poco realista).

Solo usa la libreria estandar (csv, json, random) - sin pandas.
"""
import csv
import json
import random

INPUT = "MOCK_DATA.csv"
OUTPUT = "profiles_final.csv"

# ---- Ajusta estos parametros a gusto ----
PROB_ONLINE = 0.20       # 20% de usuarios online
PROB_MODERATOR = 0.03    # 3% moderadores
MIN_TAGS, MAX_TAGS = 2, 4  # cuantos valores por usuario en interests/tags_agregados
SEED = 42                # sacala o cambiala si queres un dataset distinto cada vez
# ------------------------------------------

random.seed(SEED)


def leer_filas(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def construir_pool(filas, campo):
    """Junta todos los valores unicos que ya aparecen en esa columna del CSV."""
    return sorted({fila[campo] for fila in filas if fila[campo]})


def elegir_subset(pool, valor_base, minimo, maximo):
    """Arma una lista de 2-4 valores para un usuario, incluyendo siempre su valor original."""
    cantidad = random.randint(minimo, maximo)
    candidatos = [v for v in pool if v != valor_base]
    random.shuffle(candidatos)
    extra = candidatos[: max(0, cantidad - 1)]
    resultado = list({valor_base, *extra})
    random.shuffle(resultado)
    return resultado


def main():
    filas = leer_filas(INPUT)
    pool_interests = construir_pool(filas, "interests")
    pool_tags = construir_pool(filas, "tags_agregados")

    for fila in filas:
        fila["interests"] = json.dumps(
            elegir_subset(pool_interests, fila["interests"], MIN_TAGS, MAX_TAGS),
            ensure_ascii=False,
        )
        fila["tags_agregados"] = json.dumps(
            elegir_subset(pool_tags, fila["tags_agregados"], MIN_TAGS, MAX_TAGS),
            ensure_ascii=False,
        )
        fila["is_online"] = "true" if random.random() < PROB_ONLINE else "false"
        fila["is_moderator"] = "true" if random.random() < PROB_MODERATOR else "false"

    campos = list(filas[0].keys())
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)

    print(f"Listo: {OUTPUT} generado con {len(filas)} filas")


if __name__ == "__main__":
    main()
