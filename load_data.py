import pandas as pd
from sqlalchemy.orm import Session
from database import engine, SessionLocal
from models import Livre

# Charger les données depuis ton CSV exporté
df = pd.read_csv("livres_bruts.csv")  # 🔹 Mets ton vrai nom de fichier

def load_books():
    session: Session = SessionLocal()
    try:
        for _, row in df.iterrows():
            livre = Livre(
                titre=row.get("titre", "Sans titre"),
                description=row.get("description", ""),
                prix=float(row.get("prix", 0)),
                image=row.get("image", ""),
                disponibilite=int(row.get("stock", 1)),  # colonne stock = dispo
                note=int(row.get("note", 0)),
                page=int(row.get("page", 1))
            )
            session.add(livre)
        session.commit()
        print("✅ Les livres ont été insérés avec succès !")
    except Exception as e:
        session.rollback()
        print("❌ Erreur :", e)
    finally:
        session.close()

if __name__ == "__main__":
    load_books()
