# main.py
from fastapi import FastAPI, Depends, HTTPException, status, Request, Form, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, validator
from datetime import datetime, timedelta
from typing import Optional, List
import jwt
import os
import secrets
from email_validator import validate_email, EmailNotValidError
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)


# Configuration sécurisée
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:admin@localhost:5432/booksdb")

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de l'application
app = FastAPI(
    title="Bib Readers API",
    description="Système de gestion de bibliothèque moderne et sécurisé",
    version="1.0.0"
)

# Middleware sécurisé
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # Restreindre en production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.yourdomain.com"]
)

# Gestion des erreurs de rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configuration des templates et fichiers statiques
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration base de données
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Configuration sécurité mots de passe
pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")
security = HTTPBearer()

# Modèles SQLAlchemy
class Adherent(Base):
    __tablename__ = "adherents"
    
    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="adherent")
    date_inscription = Column(DateTime, default=datetime.utcnow)
    actif = Column(Boolean, default=True)
    tentatives_connexion = Column(Integer, default=0)
    derniere_tentative = Column(DateTime, nullable=True)
    
    emprunts = relationship("Emprunt", back_populates="adherent")
    reservations = relationship("Reservation", back_populates="adherent")
    notifications = relationship("Notification", back_populates="adherent")

class Livre(Base):
    __tablename__ = "livres"
    
    id = Column(Integer, primary_key=True, index=True)
    titre = Column(String(500), nullable=False, index=True)
    description = Column(Text)
    image_url = Column(String(500))
    prix = Column(Float)
    disponibilite = Column(Integer, default=1)
    note = Column(Integer, default=0)
    date_ajout = Column(DateTime, default=datetime.utcnow)
    
    emprunts = relationship("Emprunt", back_populates="livre")
    reservations = relationship("Reservation", back_populates="livre")

class Emprunt(Base):
    __tablename__ = "emprunts"
    
    id = Column(Integer, primary_key=True, index=True)
    id_adherent = Column(Integer, ForeignKey("adherents.id"), nullable=False)
    id_livre = Column(Integer, ForeignKey("livres.id"), nullable=False)
    date_emprunt = Column(DateTime, default=datetime.utcnow)
    date_retour_prevue = Column(DateTime, nullable=False)
    date_retour_effectif = Column(DateTime, nullable=True)
    statut = Column(String(20), default="en_cours")  # en_cours, rendu, en_retard
    
    adherent = relationship("Adherent", back_populates="emprunts")
    livre = relationship("Livre", back_populates="emprunts")

class Reservation(Base):
    __tablename__ = "reservations"
    
    id = Column(Integer, primary_key=True, index=True)
    id_adherent = Column(Integer, ForeignKey("adherents.id"), nullable=False)
    id_livre = Column(Integer, ForeignKey("livres.id"), nullable=False)
    date_reservation = Column(DateTime, default=datetime.utcnow)
    statut = Column(String(20), default="en_attente")  # en_attente, disponible, annulee
    
    adherent = relationship("Adherent", back_populates="reservations")
    livre = relationship("Livre", back_populates="reservations")

class Notification(Base):
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    id_adherent = Column(Integer, ForeignKey("adherents.id"), nullable=False)
    message = Column(Text, nullable=False)
    date_creation = Column(DateTime, default=datetime.utcnow)
    lu = Column(Boolean, default=False)
    type_notification = Column(String(50))  # rappel_retour, reservation_disponible
    
    adherent = relationship("Adherent", back_populates="notifications")

# Créer les tables
Base.metadata.create_all(bind=engine)

# Modèles Pydantic
class UserCreate(BaseModel):
    nom: str
    email: EmailStr
    password: str
    
    @validator('nom')
    def validate_nom(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Le nom doit contenir au moins 2 caractères')
        if not re.match(r"^[a-zA-ZÀ-ÿ\s'-]+$", v):
            raise ValueError('Le nom contient des caractères non autorisés')
        return v.strip()
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Le mot de passe doit contenir au moins 8 caractères')
        if not re.search(r"[A-Z]", v):
            raise ValueError('Le mot de passe doit contenir au moins une majuscule')
        if not re.search(r"[a-z]", v):
            raise ValueError('Le mot de passe doit contenir au moins une minuscule')
        if not re.search(r"[0-9]", v):
            raise ValueError('Le mot de passe doit contenir au moins un chiffre')
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError('Le mot de passe doit contenir au moins un caractère spécial')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class LivreCreate(BaseModel):
    titre: str
    description: Optional[str] = None
    prix: Optional[float] = 0.0
    disponibilite: Optional[int] = 1
    note: Optional[int] = 0

class RecommendationRequest(BaseModel):
    description: str
    
    @validator('description')
    def validate_description(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('La description doit contenir au moins 10 caractères')
        return v.strip()

# Utilitaires de sécurité
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Token invalide")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expiré")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Token invalide")
    
    user = db.query(Adherent).filter(Adherent.email == email).first()
    if user is None or not user.actif:
        raise HTTPException(status_code=401, detail="Utilisateur non trouvé ou suspendu")
    return user

def get_current_admin(current_user: Adherent = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Accès administrateur requis")
    return current_user

def is_account_locked(user: Adherent) -> bool:
    if user.tentatives_connexion >= 5:
        if user.derniere_tentative:
            lockout_time = user.derniere_tentative + timedelta(minutes=15)
            return datetime.utcnow() < lockout_time
    return False

# Fonction de recommandation
def load_recommendation_model():
    try:
        return joblib.load('recommendation_model.pkl')
    except FileNotFoundError:
        logger.warning("Modèle de recommandation non trouvé")
        return None

def get_recommendations_from_description(description: str, model_data, db: Session, n_recommendations: int = 5):
    if not model_data:
        return []
    
    try:
        # Vectoriser la description d'entrée
        description_vector = model_data['vectorizer'].transform([description])
        
        # Calculer la similarité avec tous les livres
        similarities = cosine_similarity(description_vector, model_data['tfidf_matrix']).flatten()
        
        # Obtenir les indices des livres les plus similaires
        similar_indices = similarities.argsort()[-n_recommendations-1:-1][::-1]
        
        # Récupérer les livres depuis la base de données
        livres_recommandes = []
        for idx in similar_indices:
            if idx < len(model_data['titles']):
                titre = model_data['titles'][idx]
                livre = db.query(Livre).filter(Livre.titre == titre).first()
                if livre:
                    livres_recommandes.append({
                        'id': livre.id,
                        'titre': livre.titre,
                        'description': livre.description,
                        'image_url': livre.image_url,
                        'disponibilite': livre.disponibilite,
                        'score': float(similarities[idx])
                    })
        
        return livres_recommandes
    except Exception as e:
        logger.error(f"Erreur lors des recommandations: {e}")
        return []

# Fonction d'envoi d'email
def send_notification_email(to_email: str, subject: str, body: str):
    try:
        smtp_server = os.getenv("SMTP_SERVER", "localhost")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_username = os.getenv("SMTP_USERNAME", "")
        smtp_password = os.getenv("SMTP_PASSWORD", "")
        
        msg = MIMEMultipart()
        msg['From'] = smtp_username
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        if smtp_username and smtp_password:
            server.login(smtp_username, smtp_password)
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email envoyé à {to_email}")
    except Exception as e:
        logger.error(f"Erreur envoi email: {e}")

# Routes HTML
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/inscription", response_class=HTMLResponse)
async def inscription_page(request: Request):
    return templates.TemplateResponse("inscription.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/home", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/livre/{livre_id}", response_class=HTMLResponse)
async def livre_page(request: Request, livre_id: int):
    return templates.TemplateResponse("livre.html", {"request": request, "livre_id": livre_id})

@app.get("/profil", response_class=HTMLResponse)
async def profil_page(request: Request):
    return templates.TemplateResponse("profil.html", {"request": request})

@app.get("/recommandation-par-description", response_class=HTMLResponse)
async def recommandation_page(request: Request):
    return templates.TemplateResponse("recommandation-par-description.html", {"request": request})

@app.get("/admin/gestion-adherents", response_class=HTMLResponse)
async def admin_adherents_page(request: Request):
    return templates.TemplateResponse("admin/gestion-adherents.html", {"request": request})

@app.get("/admin/gestion-livres", response_class=HTMLResponse)
async def admin_livres_page(request: Request):
    return templates.TemplateResponse("admin/gestion-livres.html", {"request": request})

@app.get("/admin/emprunts", response_class=HTMLResponse)
async def admin_emprunts_page(request: Request):
    return templates.TemplateResponse("admin/emprunts.html", {"request": request})

@app.get("/admin/statistiques", response_class=HTMLResponse)
async def admin_stats_page(request: Request):
    return templates.TemplateResponse("admin/statistiques.html", {"request": request})

# Routes API

# Authentification
@app.post("/api/register")
@limiter.limit("5/minute")
async def register(request: Request, user_data: UserCreate, db: Session = Depends(get_db)):
    # Vérifier si l'email existe déjà
    if db.query(Adherent).filter(Adherent.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Cet email est déjà utilisé")
    
    # Créer le nouvel utilisateur
    hashed_password = hash_password(user_data.password)
    new_user = Adherent(
        nom=user_data.nom,
        email=user_data.email,
        password_hash=hashed_password,
        role="adherent"
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    logger.info(f"Nouvel utilisateur créé: {user_data.email}")
    return {"message": "Compte créé avec succès", "user_id": new_user.id}

@app.post("/api/login")
@limiter.limit("10/minute")
async def login(request: Request, user_credentials: UserLogin, db: Session = Depends(get_db)):
    user = db.query(Adherent).filter(Adherent.email == user_credentials.email).first()
    
    if not user:
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")
    
    # Vérifier si le compte est verrouillé
    if is_account_locked(user):
        raise HTTPException(status_code=429, detail="Compte temporairement verrouillé. Réessayez dans 15 minutes.")
    
    # Vérifier le mot de passe
    if not verify_password(user_credentials.password, user.password_hash):
        user.tentatives_connexion += 1
        user.derniere_tentative = datetime.utcnow()
        db.commit()
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")
    
    if not user.actif:
        raise HTTPException(status_code=401, detail="Compte suspendu")
    
    # Réinitialiser les tentatives de connexion
    user.tentatives_connexion = 0
    user.derniere_tentative = None
    db.commit()
    
    # Créer le token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "role": user.role}, expires_delta=access_token_expires
    )
    
    logger.info(f"Connexion réussie: {user.email}")
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "nom": user.nom,
            "email": user.email,
            "role": user.role
        }
    }

# Gestion des livres
@app.get("/api/livres")
async def get_livres(search: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Livre)
    
    if search:
        query = query.filter(Livre.titre.ilike(f"%{search}%"))
    
    livres = query.all()
    return [
        {
            "id": livre.id,
            "titre": livre.titre,
            "description": livre.description,
            "image_url": livre.image_url,
            "disponibilite": livre.disponibilite,
            "note": livre.note,
            "prix": livre.prix
        }
        for livre in livres
    ]

@app.get("/api/livre/{livre_id}")
async def get_livre_detail(livre_id: int, db: Session = Depends(get_db)):
    livre = db.query(Livre).filter(Livre.id == livre_id).first()
    if not livre:
        raise HTTPException(status_code=404, detail="Livre non trouvé")
    
    return {
        "id": livre.id,
        "titre": livre.titre,
        "description": livre.description,
        "image_url": livre.image_url,
        "disponibilite": livre.disponibilite,
        "note": livre.note,
        "prix": livre.prix
    }

@app.post("/api/livres")
async def create_livre(
    livre_data: LivreCreate,
    current_user: Adherent = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    new_livre = Livre(**livre_data.dict())
    db.add(new_livre)
    db.commit()
    db.refresh(new_livre)
    
    logger.info(f"Nouveau livre ajouté: {livre_data.titre} par {current_user.email}")
    return {"message": "Livre ajouté avec succès", "livre_id": new_livre.id}

@app.put("/api/livres/{livre_id}")
async def update_livre(
    livre_id: int,
    livre_data: LivreCreate,
    current_user: Adherent = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    livre = db.query(Livre).filter(Livre.id == livre_id).first()
    if not livre:
        raise HTTPException(status_code=404, detail="Livre non trouvé")
    
    for key, value in livre_data.dict().items():
        setattr(livre, key, value)
    
    db.commit()
    logger.info(f"Livre modifié: {livre_id} par {current_user.email}")
    return {"message": "Livre mis à jour avec succès"}

@app.delete("/api/livres/{livre_id}")
async def delete_livre(
    livre_id: int,
    current_user: Adherent = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    livre = db.query(Livre).filter(Livre.id == livre_id).first()
    if not livre:
        raise HTTPException(status_code=404, detail="Livre non trouvé")
    
    # Vérifier s'il y a des emprunts en cours
    emprunts_actifs = db.query(Emprunt).filter(
        Emprunt.id_livre == livre_id,
        Emprunt.statut == "en_cours"
    ).count()
    
    if emprunts_actifs > 0:
        raise HTTPException(status_code=400, detail="Impossible de supprimer un livre avec des emprunts en cours")
    
    db.delete(livre)
    db.commit()
    
    logger.info(f"Livre supprimé: {livre_id} par {current_user.email}")
    return {"message": "Livre supprimé avec succès"}

# Réservations
@app.post("/api/reservations")
async def create_reservation(
    livre_id: int,
    current_user: Adherent = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Vérifier si le livre existe
    livre = db.query(Livre).filter(Livre.id == livre_id).first()
    if not livre:
        raise HTTPException(status_code=404, detail="Livre non trouvé")
    
    # Vérifier si l'utilisateur a déjà réservé ce livre
    existing_reservation = db.query(Reservation).filter(
        Reservation.id_adherent == current_user.id,
        Reservation.id_livre == livre_id,
        Reservation.statut == "en_attente"
    ).first()
    
    if existing_reservation:
        raise HTTPException(status_code=400, detail="Vous avez déjà réservé ce livre")
    
    # Créer la réservation
    new_reservation = Reservation(
        id_adherent=current_user.id,
        id_livre=livre_id,
        statut="en_attente"
    )
    
    db.add(new_reservation)
    db.commit()
    
    logger.info(f"Réservation créée: livre {livre_id} par {current_user.email}")
    return {"message": "Réservation créée avec succès"}

# Emprunts
@app.get("/api/mes-emprunts")
async def get_mes_emprunts(
    current_user: Adherent = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    emprunts = db.query(Emprunt).filter(Emprunt.id_adherent == current_user.id).all()
    
    result = []
    for emprunt in emprunts:
        livre = db.query(Livre).filter(Livre.id == emprunt.id_livre).first()
        
        # Vérifier si en retard
        en_retard = False
        if emprunt.statut == "en_cours" and datetime.utcnow() > emprunt.date_retour_prevue:
            en_retard = True
            # Mettre à jour le statut si nécessaire
            emprunt.statut = "en_retard"
            db.commit()
        
        result.append({
            "id": emprunt.id,
            "livre": {
                "id": livre.id,
                "titre": livre.titre,
                "image_url": livre.image_url
            },
            "date_emprunt": emprunt.date_emprunt.isoformat(),
            "date_retour_prevue": emprunt.date_retour_prevue.isoformat(),
            "date_retour_effectif": emprunt.date_retour_effectif.isoformat() if emprunt.date_retour_effectif else None,
            "statut": emprunt.statut,
            "en_retard": en_retard
        })
    
    return result

@app.post("/api/emprunts")
async def create_emprunt(
    id_adherent: int,
    id_livre: int,
    current_user: Adherent = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    # Vérifier si le livre est disponible
    livre = db.query(Livre).filter(Livre.id == id_livre).first()
    if not livre or livre.disponibilite <= 0:
        raise HTTPException(status_code=400, detail="Livre non disponible")
    
    # Vérifier si l'adhérent existe
    adherent = db.query(Adherent).filter(Adherent.id == id_adherent).first()
    if not adherent:
        raise HTTPException(status_code=404, detail="Adhérent non trouvé")
    
    # Créer l'emprunt
    date_retour_prevue = datetime.utcnow() + timedelta(days=14)
    new_emprunt = Emprunt(
        id_adherent=id_adherent,
        id_livre=id_livre,
        date_retour_prevue=date_retour_prevue
    )
    
    # Diminuer la disponibilité
    livre.disponibilite -= 1
    
    db.add(new_emprunt)
    db.commit()
    
    logger.info(f"Emprunt créé: livre {id_livre} par adhérent {id_adherent}")
    return {"message": "Emprunt enregistré avec succès"}

@app.post("/api/retours/{emprunt_id}")
async def create_retour(
    emprunt_id: int,
    current_user: Adherent = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    emprunt = db.query(Emprunt).filter(Emprunt.id == emprunt_id).first()
    if not emprunt:
        raise HTTPException(status_code=404, detail="Emprunt non trouvé")
    
    if emprunt.date_retour_effectif:
        raise HTTPException(status_code=400, detail="Livre déjà retourné")
    
    # Marquer comme retourné
    emprunt.date_retour_effectif = datetime.utcnow()
    emprunt.statut = "rendu"
    
    # Augmenter la disponibilité
    livre = db.query(Livre).filter(Livre.id == emprunt.id_livre).first()
    livre.disponibilite += 1
    
    db.commit()
    
    logger.info(f"Retour enregistré: emprunt {emprunt_id}")
    return {"message": "Retour enregistré avec succès"}

# Recommandations
@app.post("/api/recommander-par-description")
async def get_recommendations(
    request_data: RecommendationRequest,
    db: Session = Depends(get_db)
):
    model_data = load_recommendation_model()
    if not model_data:
        raise HTTPException(status_code=503, detail="Service de recommandation indisponible")
    
    recommendations = get_recommendations_from_description(
        request_data.description, model_data, db
    )
    
    return {"recommendations": recommendations}

# Gestion des adhérents (Admin)
@app.get("/api/adherents")
async def get_adherents(
    current_user: Adherent = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    adherents = db.query(Adherent).all()
    return [
        {
            "id": adherent.id,
            "nom": adherent.nom,
            "email": adherent.email,
            "role": adherent.role,
            "actif": adherent.actif,
            "date_inscription": adherent.date_inscription.isoformat()
        }
        for adherent in adherents
    ]

@app.put("/api/adherents/{adherent_id}")
async def update_adherent(
    adherent_id: int,
    nom: Optional[str] = None,
    role: Optional[str] = None,
    actif: Optional[bool] = None,
    current_user: Adherent = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    adherent = db.query(Adherent).filter(Adherent.id == adherent_id).first()
    if not adherent:
        raise HTTPException(status_code=404, detail="Adhérent non trouvé")
    
    if nom:
        adherent.nom = nom
    if role:
        adherent.role = role
    if actif is not None:
        adherent.actif = actif
    
    db.commit()
    
    logger.info(f"Adhérent modifié: {adherent_id} par {current_user.email}")
    return {"message": "Adhérent mis à jour avec succès"}

@app.delete("/api/adherents/{adherent_id}")
async def suspend_adherent(
    adherent_id: int,
    current_user: Adherent = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    if adherent_id == current_user.id:
        raise HTTPException(status_code=400, detail="Vous ne pouvez pas vous suspendre vous-même")
    
    adherent = db.query(Adherent).filter(Adherent.id == adherent_id).first()
    if not adherent:
        raise HTTPException(status_code=404, detail="Adhérent non trouvé")
    
    adherent.actif = False
    db.commit()
    
    logger.info(f"Adhérent suspendu: {adherent_id} par {current_user.email}")
    return {"message": "Adhérent suspendu avec succès"}

# Statistiques
@app.get("/api/statistiques")
async def get_statistiques(
    current_user: Adherent = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    # Top 5 livres les plus empruntés
    top_livres = db.query(
        Livre.titre,
        db.func.count(Emprunt.id).label('nb_emprunts')
    ).join(Emprunt).group_by(Livre.id).order_by(
        db.func.count(Emprunt.id).desc()
    ).limit(5).all()
    
    # Taux de disponibilité global
    total_livres = db.query(Livre).count()
    livres_disponibles = db.query(Livre).filter(Livre.disponibilite > 0).count()
    taux_disponibilite = (livres_disponibles / total_livres * 100) if total_livres > 0 else 0
    
    # Nombre de retards
    nb_retards = db.query(Emprunt).filter(
        Emprunt.statut == "en_retard"
    ).count()
    
    # Emprunts par mois
    emprunts_par_mois = db.query(
        db.func.date_part('month', Emprunt.date_emprunt).label('mois'),
        db.func.count(Emprunt.id).label('nb_emprunts')
    ).group_by(db.func.date_part('month', Emprunt.date_emprunt)).all()
    
    return {
        "top_livres": [{"titre": livre.titre, "nb_emprunts": livre.nb_emprunts} for livre in top_livres],
        "taux_disponibilite": round(taux_disponibilite, 2),
        "nb_retards": nb_retards,
        "emprunts_par_mois": [{"mois": int(emp.mois), "nb_emprunts": emp.nb_emprunts} for emp in emprunts_par_mois],
        "total_adherents": db.query(Adherent).count(),
        "total_livres": total_livres,
        "emprunts_actifs": db.query(Emprunt).filter(Emprunt.statut == "en_cours").count()
    }

# Notifications
@app.get("/api/notifications")
async def get_notifications(
    current_user: Adherent = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    notifications = db.query(Notification).filter(
        Notification.id_adherent == current_user.id
    ).order_by(Notification.date_creation.desc()).all()
    
    return [
        {
            "id": notif.id,
            "message": notif.message,
            "date_creation": notif.date_creation.isoformat(),
            "lu": notif.lu,
            "type_notification": notif.type_notification
        }
        for notif in notifications
    ]

@app.put("/api/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: int,
    current_user: Adherent = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    notification = db.query(Notification).filter(
        Notification.id == notification_id,
        Notification.id_adherent == current_user.id
    ).first()
    
    if not notification:
        raise HTTPException(status_code=404, detail="Notification non trouvée")
    
    notification.lu = True
    db.commit()
    
    return {"message": "Notification marquée comme lue"}

# Tâche de notification automatique (à exécuter périodiquement)
@app.post("/api/send-reminders")
async def send_reminders(
    current_user: Adherent = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    # Rappels de retour (2 jours avant échéance)
    date_limite = datetime.utcnow() + timedelta(days=2)
    emprunts_a_rappeler = db.query(Emprunt).filter(
        Emprunt.statut == "en_cours",
        Emprunt.date_retour_prevue <= date_limite,
        Emprunt.date_retour_prevue > datetime.utcnow()
    ).all()
    
    rappels_envoyes = 0
    for emprunt in emprunts_a_rappeler:
        adherent = db.query(Adherent).filter(Adherent.id == emprunt.id_adherent).first()
        livre = db.query(Livre).filter(Livre.id == emprunt.id_livre).first()
        
        # Créer notification
        notification = Notification(
            id_adherent=adherent.id,
            message=f"Rappel: Le livre '{livre.titre}' doit être retourné le {emprunt.date_retour_prevue.strftime('%d/%m/%Y')}",
            type_notification="rappel_retour"
        )
        db.add(notification)
        
        # Envoyer email
        send_notification_email(
            adherent.email,
            "Rappel de retour de livre",
            f"""
            <h2>Rappel de retour</h2>
            <p>Bonjour {adherent.nom},</p>
            <p>Ce message vous rappelle que le livre <strong>"{livre.titre}"</strong> doit être retourné le <strong>{emprunt.date_retour_prevue.strftime('%d/%m/%Y')}</strong>.</p>
            <p>Merci de penser à le retourner à temps.</p>
            <p>L'équipe Bib Readers</p>
            """
        )
        
        rappels_envoyes += 1
    
    db.commit()
    
    return {"message": f"{rappels_envoyes} rappels envoyés"}

# Middleware de sécurité supplémentaire
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline' cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' cdnjs.cloudflare.com"
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)