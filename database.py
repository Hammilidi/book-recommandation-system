from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config import settings

# moteur PostgreSQL
engine = create_engine(settings.DATABASE_URL)  

# session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# base modèle
Base = declarative_base()
