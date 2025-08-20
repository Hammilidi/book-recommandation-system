from fastapi import FastAPI, Depends, HTTPException, status, Request, Form, Response, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional, List
import secrets
import os
import re
import logging
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from jose import jwt as jose_jwt, JWTError
from email_validator import validate_email, EmailNotValidError

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1"]
)

# Gestion des erreurs de rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configuration des templates et fichiers statiques
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration base de données
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Configuration sécurité mots de passe
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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
    
    emprunts = relationship("Emprunt", back_populates="adherent")
    reservations = relationship("Reservation", back_populates="adherent")
    notifications = relationship("Notification", back_populates="adherent")

class Livre(Base):
    __tablename__ = "livres"
    
    id = Column(Integer, primary_key=True, index=True)
    titre = Column(String(500), nullable=False, index=True)
    description = Column(Text)
    prix = Column(Float)
    image_url = Column(String(500))
    disponibilite = Column(Integer, default=1)
    note = Column(Integer, default=0)
    
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
    statut = Column(String(20), default="en_cours")
    
    adherent = relationship("Adherent", back_populates="emprunts")
    livre = relationship("Livre", back_populates="emprunts")

class Reservation(Base):
    __tablename__ = "reservations"
    
    id = Column(Integer, primary_key=True, index=True)
    id_adherent = Column(Integer, ForeignKey("adherents.id"), nullable=False)
    id_livre = Column(Integer, ForeignKey("livres.id"), nullable=False)
    date_reservation = Column(DateTime, default=datetime.utcnow)
    statut = Column(String(20), default="en_attente")
    
    adherent = relationship("Adherent", back_populates="reservations")
    livre = relationship("Livre", back_populates="reservations")

class HistoriqueEmprunt(Base):
    __tablename__ = "historique_emprunts"
    id = Column(Integer, primary_key=True)
    id_adherent = Column(Integer, nullable=False)
    id_livre = Column(Integer, nullable=False)
    date_emprunt = Column(DateTime, default=datetime.utcnow)

class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True, index=True)
    id_adherent = Column(Integer, ForeignKey("adherents.id"), nullable=False)
    message = Column(Text, nullable=False)
    date_creation = Column(DateTime, default=datetime.utcnow)
    lu = Column(Boolean, default=False)
    
    adherent = relationship("Adherent", back_populates="notifications")

# Créer les tables (à exécuter une fois)
Base.metadata.create_all(bind=engine)

# Modèles Pydantic
class UserCreate(BaseModel):
    nom: str
    email: EmailStr
    password: str
    password_confirm: str

    @property
    def password_match(self):
        return self.password == self.password_confirm

class AdherentOut(BaseModel):
    id: int
    nom: str
    email: str
    role: str
    date_inscription: datetime
    actif: bool

    class Config:
        from_attributes = True

class LivreOut(BaseModel):
    id: int
    titre: str
    description: Optional[str]
    image_url: Optional[str]
    disponibilite: int
    note: Optional[int]

    class Config:
        from_attributes = True

class EmpruntOut(BaseModel):
    id: int
    livre_titre: str
    date_emprunt: datetime
    date_retour_prevue: datetime
    statut: str

class DescriptionRequest(BaseModel):
    description: str

# Fonctions de sécurité
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jose_jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

def get_current_user(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token manquant")
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalide")
    user_id = payload.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalide")
    user = db.query(Adherent).filter(Adherent.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Utilisateur non trouvé")
    return user

def get_current_admin(user: Adherent = Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Accès refusé. Administrateur requis.")
    return user

# =================================================================
#                         LOGIQUE DE RECOMMANDATION
# =================================================================
# Variables globales pour le modèle de recommandation
tfidf_vectorizer = None
tfidf_matrix = None
livres_df = None

def preprocess_text(text):
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_or_train_model():
    global tfidf_vectorizer, tfidf_matrix, livres_df
    model_path = "models/recommendation_model.joblib"

    if os.path.exists(model_path):
        try:
            model_data = joblib.load(model_path)
            tfidf_vectorizer = model_data['tfidf_vectorizer']
            tfidf_matrix = model_data['tfidf_matrix']
            livres_df = model_data['df']
            logger.info("Modèles de recommandation chargés depuis le disque.")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles sauvegardés : {e}")
            pass

    logger.warning("Modèles de recommandation non trouvés ou erreur. Entraînement en cours...")
    try:
        db = SessionLocal()
        livres = db.query(Livre).filter(Livre.description.isnot(None), Livre.description != "").all()
        livres_df = pd.DataFrame([(l.id, l.titre, l.description, l.image_url, l.disponibilite, l.note) for l in livres],
                                 columns=['id', 'titre', 'description', 'image_url', 'disponibilite', 'note'])
        db.close()
        
        if livres_df.empty:
            logger.error("Aucun livre avec une description trouvée dans la base de données. Impossible d'entraîner le modèle.")
            return False

        livres_df['description_clean'] = livres_df['description'].apply(preprocess_text)
        tfidf_vectorizer = TfidfVectorizer(stop_words='french')
        tfidf_matrix = tfidf_vectorizer.fit_transform(livres_df['description_clean'])

        if not os.path.exists("models"):
            os.makedirs("models")
        
        model_data = {
            'tfidf_vectorizer': tfidf_vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'df': livres_df
        }
        joblib.dump(model_data, model_path)
        logger.info("Modèles de recommandation entraînés et sauvegardés.")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement du modèle : {e}")
        return False

# Initialiser les modèles au démarrage de l'application
@app.on_event("startup")
async def startup_event():
    load_or_train_model()

# =================================================================
#                         ROUTES HTML
# =================================================================

@app.get("/", response_class=HTMLResponse)
@limiter.limit("5/minute")
async def read_root(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/inscription", response_class=HTMLResponse)
@limiter.limit("5/minute")
async def inscription_page(request: Request):
    return templates.TemplateResponse("inscription.html", {"request": request})

@app.get("/home", response_class=HTMLResponse)
async def home_page(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    livres = db.query(Livre).order_by(Livre.titre).all()
    return templates.TemplateResponse("home.html", {"request": request, "user": user, "livres": livres})

@app.get("/livre/{livre_id}", response_class=HTMLResponse)
async def livre_detail_page(request: Request, livre_id: int, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    livre = db.query(Livre).filter(Livre.id == livre_id).first()
    if not livre:
        raise HTTPException(status_code=404, detail="Livre non trouvé")
    return templates.TemplateResponse("livre.html", {"request": request, "user": user, "livre": livre})

@app.get("/profil", response_class=HTMLResponse)
async def profil_page(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    emprunts = db.query(Emprunt).filter(Emprunt.id_adherent == user.id).all()
    reservations = db.query(Reservation).filter(Reservation.id_adherent == user.id).all()
    return templates.TemplateResponse("profil.html", {"request": request, "user": user, "emprunts": emprunts, "reservations": reservations})

@app.get("/mes-emprunts", response_class=HTMLResponse)
async def mes_emprunts_page(
    request: Request,
    db: Session = Depends(get_db),
    user: Adherent = Depends(get_current_user)
):
    # On récupère les emprunts de l’utilisateur connecté
    emprunts = db.query(Emprunt).filter(Emprunt.id_adherent == user.id).all()
    
    # On prépare les données pour le template
    emprunts_out = []
    for emprunt in emprunts:
        emprunts_out.append({
            "id": emprunt.id,
            "livre_titre": emprunt.livre.titre,
            "date_emprunt": emprunt.date_emprunt,
            "date_retour_prevue": emprunt.date_retour_prevue,
            "date_retour_effectif": emprunt.date_retour_effectif,
            "statut": "en retard" if emprunt.date_retour_prevue < datetime.utcnow() and not emprunt.date_retour_effectif else emprunt.statut
        })

    # On envoie les données au template
    return templates.TemplateResponse(
        "mes_emprunts.html",
        {"request": request, "user": user, "emprunts": emprunts_out}
    )

@app.get("/recommandations", response_class=HTMLResponse)
async def recommandations_page(request: Request, user: Adherent = Depends(get_current_user)):
    return templates.TemplateResponse("recommandation_par_description.html", {"request": request, "user": user})

@app.get("/admin/gestion-adherents", response_class=HTMLResponse)
async def admin_adherents_page(request: Request, db: Session = Depends(get_db), admin: Adherent = Depends(get_current_admin)):
    adherents = db.query(Adherent).all()
    return templates.TemplateResponse("admin/gestion_adherents.html", {"request": request, "adherents": adherents, "user": admin})

@app.get("/admin/gestion-livres", response_class=HTMLResponse)
async def admin_livres_page(request: Request, db: Session = Depends(get_db), admin: Adherent = Depends(get_current_admin)):
    livres = db.query(Livre).all()
    return templates.TemplateResponse("admin/gestion_livres.html", {"request": request, "livres": livres, "user": admin})

@app.get("/admin/emprunts", response_class=HTMLResponse)
async def admin_emprunts_page(request: Request, db: Session = Depends(get_db), admin: Adherent = Depends(get_current_admin)):
    adherents = db.query(Adherent).all()
    livres = db.query(Livre).all()
    emprunts = db.query(Emprunt).all()
    return templates.TemplateResponse("admin/emprunts.html", {"request": request, "adherents": adherents, "livres": livres, "emprunts": emprunts, "user": admin})

@app.get("/admin/statistiques", response_class=HTMLResponse)
async def admin_stats_page(request: Request, admin: Adherent = Depends(get_current_admin)):
    return templates.TemplateResponse("admin/statistiques.html", {"request": request, "user": admin})

@app.get("/logout")
async def logout(response: Response):
    response.delete_cookie(key="access_token")
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

# =================================================================
#                         ROUTES API
# =================================================================

@app.post("/api/register", status_code=status.HTTP_201_CREATED, response_model=AdherentOut)
@limiter.limit("5/minute")
def register_user(request: Request, user: UserCreate, db: Session = Depends(get_db)):
    if not user.password_match:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Les mots de passe ne correspondent pas")
    
    try:
        validate_email(user.email)
    except EmailNotValidError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Adresse e-mail invalide")
    
    if db.query(Adherent).filter(Adherent.email == user.email).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cette adresse e-mail est déjà utilisée")

    hashed_password = pwd_context.hash(user.password)
    new_user = Adherent(
        nom=user.nom,
        email=user.email,
        password_hash=hashed_password,
        role="adherent"
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# Nouvelle route pour créer un utilisateur avec le rôle "admin"
@app.post("/api/admin/create", status_code=status.HTTP_201_CREATED, response_model=AdherentOut)
def create_admin_user(request: Request, user: UserCreate, db: Session = Depends(get_db)):
    # Vérifier si un administrateur existe déjà dans la base de données
    if db.query(Adherent).filter(Adherent.role == "admin").first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Un administrateur existe déjà.")
    
    if not user.password_match:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Les mots de passe ne correspondent pas")
    
    try:
        validate_email(user.email)
    except EmailNotValidError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Adresse e-mail invalide")
    
    # Vérifier si l'email est déjà utilisé par un autre adhérent
    if db.query(Adherent).filter(Adherent.email == user.email).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cette adresse e-mail est déjà utilisée")

    hashed_password = pwd_context.hash(user.password)
    new_admin = Adherent(
        nom=user.nom,
        email=user.email,
        password_hash=hashed_password,
        role="admin"  # Définir le rôle comme "admin"
    )
    db.add(new_admin)
    db.commit()
    db.refresh(new_admin)
    return new_admin

@app.post("/api/login", response_class=JSONResponse)
@limiter.limit("5/minute")
def login(request: Request, response: Response, db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    user = db.query(Adherent).filter(Adherent.email == form_data.username).first()
    if not user or not pwd_context.verify(form_data.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Identifiants incorrects")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"user_id": user.id, "role": user.role}, expires_delta=access_token_expires
    )
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return {"message": "Connexion réussie"}

@app.get("/api/livres", response_model=List[LivreOut])
def get_livres(search: Optional[str] = Query(None), db: Session = Depends(get_db)):
    query = db.query(Livre)
    if search:
        query = query.filter(Livre.titre.ilike(f"%{search}%"))
    return query.all()

@app.post("/api/reservations/{livre_id}")
def reserver_livre(livre_id: int, db: Session = Depends(get_db), user: Adherent = Depends(get_current_user)):
    livre = db.query(Livre).filter(Livre.id == livre_id).first()
    if not livre:
        raise HTTPException(status_code=404, detail="Livre non trouvé")
    
    if livre.disponibilite <= 0:
        raise HTTPException(status_code=400, detail="Ce livre n'est pas disponible pour le moment")
    
    existing_reservation = db.query(Reservation).filter(
        Reservation.id_adherent == user.id,
        Reservation.id_livre == livre.id,
        Reservation.statut == "en_attente"
    ).first()
    if existing_reservation:
        raise HTTPException(status_code=400, detail="Vous avez déjà une réservation en cours pour ce livre")
    
    new_reservation = Reservation(
        id_adherent=user.id,
        id_livre=livre.id,
        date_reservation=datetime.utcnow(),
        statut="en_attente"
    )
    livre.disponibilite -= 1
    db.add(new_reservation)
    db.commit()
    
    return {"message": "Réservation effectuée avec succès"}

@app.post("/api/adherents/delete/{id}")
def delete_adherent(id: int, db: Session = Depends(get_db), admin: Adherent = Depends(get_current_admin)):
    adherent = db.query(Adherent).filter(Adherent.id == id).first()
    if not adherent:
        raise HTTPException(status_code=404, detail="Adhérent non trouvé")
    db.delete(adherent)
    db.commit()
    return {"message": "Adhérent supprimé"}

@app.post("/api/adherents/suspend/{id}")
def suspend_adherent(id: int, db: Session = Depends(get_db), admin: Adherent = Depends(get_current_admin)):
    adherent = db.query(Adherent).filter(Adherent.id == id).first()
    if not adherent:
        raise HTTPException(status_code=404, detail="Adhérent non trouvé")
    adherent.actif = False
    db.commit()
    return {"message": "Adhérent suspendu"}

@app.get("/api/mes-emprunts", response_model=List[EmpruntOut])
def get_mes_emprunts(db: Session = Depends(get_db), user: Adherent = Depends(get_current_user)):
    emprunts = db.query(Emprunt).filter(Emprunt.id_adherent == user.id).all()
    emprunts_out = []
    for emprunt in emprunts:
        emprunt_out = EmpruntOut(
            id=emprunt.id,
            livre_titre=emprunt.livre.titre,
            date_emprunt=emprunt.date_emprunt,
            date_retour_prevue=emprunt.date_retour_prevue,
            statut="en retard" if emprunt.date_retour_prevue < datetime.utcnow() and not emprunt.date_retour_effectif else emprunt.statut
        )
        emprunts_out.append(emprunts_out)
    return emprunts_out

@app.post("/api/recommander-par-description", response_model=List[LivreOut])
def recommander_par_description(request_body: DescriptionRequest, db: Session = Depends(get_db), user: Adherent = Depends(get_current_user)):
    description = request_body.description
    if not description:
        raise HTTPException(status_code=400, detail="Veuillez fournir une description")
    
    if tfidf_vectorizer is None or tfidf_matrix is None or livres_df is None:
        raise HTTPException(status_code=503, detail="Le modèle de recommandation n'a pas pu être chargé ou entraîné.")

    try:
        user_description_vector = tfidf_vectorizer.transform([preprocess_text(description)])
        cosine_similarities = cosine_similarity(user_description_vector, tfidf_matrix).flatten()
        
        top_indices = cosine_similarities.argsort()[-5:][::-1]
        
        livres_recommandes = []
        livres_ids = livres_df.loc[top_indices, 'id'].tolist()
        
        for livre_id in livres_ids:
            livre = db.query(Livre).filter(Livre.id == livre_id).first()
            if livre:
                livres_recommandes.append(livre)
                
        return livres_recommandes
    except Exception as e:
        logger.error(f"Erreur lors de la recommandation: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur lors de la recommandation")

@app.get("/api/statistiques")
def get_statistiques(db: Session = Depends(get_db), admin: Adherent = Depends(get_current_admin)):
    total_livres = db.query(Livre).count()
    livres_disponibles = db.query(Livre).filter(Livre.disponibilite > 0).count()
    taux_disponibilite = (livres_disponibles / total_livres) * 100 if total_livres > 0 else 0
    
    livres_populaires = (
        db.query(Livre.titre, db.func.count(Emprunt.id))
        .join(Emprunt)
        .group_by(Livre.titre)
        .order_by(db.func.count(Emprunt.id).desc())
        .limit(5)
        .all()
    )
    
    retards = db.query(Emprunt).filter(
        Emprunt.date_retour_prevue < datetime.utcnow(),
        Emprunt.date_retour_effectif.is_(None)
    ).count()
    
    return {
        "taux_disponibilite": round(taux_disponibilite, 2),
        "livres_plus_empruntes": [{"titre": l[0], "emprunts": l[1]} for l in livres_populaires],
        "nombre_retards": retards
    }
    
@app.post("/api/livres")
def add_livre(titre: str = Form(...), auteur: str = Form(...), description: str = Form(...), image_url: str = Form(...), disponibilite: int = Form(...), db: Session = Depends(get_db), admin: Adherent = Depends(get_current_admin)):
    new_livre = Livre(titre=titre, auteur=auteur, description=description, image_url=image_url, disponibilite=disponibilite)
    db.add(new_livre)
    db.commit()
    # Mettre à jour le modèle de recommandation après l'ajout d'un livre
    load_or_train_model()
    return {"message": "Livre ajouté avec succès"}

@app.post("/api/livres/delete/{livre_id}")
def delete_livre(livre_id: int, db: Session = Depends(get_db), admin: Adherent = Depends(get_current_admin)):
    livre = db.query(Livre).filter(Livre.id == livre_id).first()
    if not livre:
        raise HTTPException(status_code=404, detail="Livre non trouvé")
    db.delete(livre)
    db.commit()
    # Mettre à jour le modèle de recommandation après la suppression d'un livre
    load_or_train_model()
    return {"message": "Livre supprimé"}

@app.post("/api/livres/update/{livre_id}")
def update_livre(livre_id: int, titre: str = Form(...), auteur: str = Form(...), description: str = Form(...), image_url: str = Form(...), disponibilite: int = Form(...), db: Session = Depends(get_db), admin: Adherent = Depends(get_current_admin)):
    livre = db.query(Livre).filter(Livre.id == livre_id).first()
    if not livre:
        raise HTTPException(status_code=404, detail="Livre non trouvé")
    livre.titre = titre
    livre.auteur = auteur
    livre.description = description
    livre.image_url = image_url
    livre.disponibilite = disponibilite
    db.commit()
    # Mettre à jour le modèle de recommandation après la modification d'un livre
    load_or_train_model()
    return {"message": "Livre mis à jour"}

@app.post("/api/emprunts")
def emprunter_livre(id_adherent: int = Form(...), id_livre: int = Form(...), date_retour_prevue: str = Form(...), db: Session = Depends(get_db), admin: Adherent = Depends(get_current_admin)):
    livre = db.query(Livre).filter(Livre.id == id_livre).first()
    if not livre or livre.disponibilite < 1:
        raise HTTPException(status_code=400, detail="Livre non disponible")
    
    date_prevue = datetime.strptime(date_retour_prevue, '%Y-%m-%d')
    new_emprunt = Emprunt(id_adherent=id_adherent, id_livre=id_livre, date_retour_prevue=date_prevue)
    livre.disponibilite -= 1
    db.add(new_emprunt)
    db.commit()
    return {"message": "Emprunt enregistré"}

@app.post("/api/retours/{emprunt_id}")
def retour_livre(emprunt_id: int, db: Session = Depends(get_db), admin: Adherent = Depends(get_current_admin)):
    emprunt = db.query(Emprunt).filter(Emprunt.id == emprunt_id).first()
    if not emprunt or emprunt.date_retour_effectif:
        raise HTTPException(status_code=400, detail="Emprunt invalide ou déjà retourné")
    
    emprunt.date_retour_effectif = datetime.utcnow()
    emprunt.statut = "retourne"
    emprunt.livre.disponibilite += 1
    db.commit()
    return {"message": "Retour enregistré"}
