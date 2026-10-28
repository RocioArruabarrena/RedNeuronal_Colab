import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app.database import get_db

db = next(get_db())
filas = db.execute(
    text(
        "SELECT user_id, friend_id, real_label, predicted_score "
        "FROM feedback ORDER BY user_id, friend_id"
    )
).fetchall()

print("user_id | friend_id | real_label | predicted_score")
for f in filas:
    print(f"{f[0]:>7} | {f[1]:>9} | {str(f[2]):>10} | {f[3]:.4f}")
print(f"\nTotal: {len(filas)} feedbacks")
