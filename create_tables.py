from database import Base, engine
import models  # importe tous les modèles pour que SQLAlchemy les connaisse

Base.metadata.create_all(bind=engine)
print("Tables créées")
