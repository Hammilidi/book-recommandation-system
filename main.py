from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Adherent, Livre, Emprunt, Reservation, Notification
from passlib.hash import bcrypt
import joblib
from datetime import datetime
from sqlalchemy import func
# --- App et templates ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Dépendance DB ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------- AUTHENTIFICATION --------------------

@app.get("/inscription", response_class=HTMLResponse)
def page_inscription(request: Request):
    return templates.TemplateResponse("inscription.html", {"request": request})

@app.post("/api/register")
def register(
    request: Request,
    nom: str = Form(...),
    email: str = Form(...),
    motdepasse: str = Form(...),
    confirm: str = Form(...),
    db: Session = Depends(get_db)
):
    if motdepasse != confirm:
        return templates.TemplateResponse("inscription.html", {"request": request, "error": "Les mots de passe ne correspondent pas."})
    if db.query(Adherent).filter(Adherent.email == email).first():
        return templates.TemplateResponse("inscription.html", {"request": request, "error": "Cet email existe déjà."})
    user = Adherent(nom=nom, email=email, password_hash=bcrypt.hash(motdepasse), role="adherent")
    db.add(user)
    db.commit()
    return RedirectResponse("/login", status_code=303)

@app.get("/login", response_class=HTMLResponse)
def page_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/api/login")
def login(
    request: Request,
    email: str = Form(...),
    motdepasse: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(Adherent).filter(Adherent.email == email).first()
    if not user or not bcrypt.verify(motdepasse, user.password_hash):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Identifiants incorrects."})
    if user.role == "admin":
        return RedirectResponse("/admin/dashboard", status_code=303)
    return RedirectResponse("/home", status_code=303)

# -------------------- ADHERENT --------------------

@app.get("/home", response_class=HTMLResponse)
def page_home(request: Request, search: str = "", db: Session = Depends(get_db)):
    livres = db.query(Livre).filter(Livre.titre.ilike(f"%{search}%")).all()
    return templates.TemplateResponse("home.html", {"request": request, "livres": livres})

@app.get("/livre/{id}", response_class=HTMLResponse)
def page_livre(request: Request, id: int, db: Session = Depends(get_db)):
    livre = db.query(Livre).get(id)
    return templates.TemplateResponse("livre.html", {"request": request, "livre": livre})

@app.post("/api/reservations")
def reserver_livre(id_adherent: int = Form(...), id_livre: int = Form(...), db: Session = Depends(get_db)):
    livre = db.query(Livre).get(id_livre)
    if livre.disponibilite < 1:
        raise HTTPException(status_code=400, detail="Livre non disponible")
    if db.query(Reservation).filter_by(id_adherent=id_adherent, id_livre=id_livre, statut="en_attente").first():
        raise HTTPException(status_code=400, detail="Vous avez déjà réservé ce livre")
    reservation = Reservation(id_adherent=id_adherent, id_livre=id_livre)
    livre.disponibilite -= 1
    db.add(reservation)
    db.commit()
    return {"message": "Réservation réussie"}

@app.get("/profil", response_class=HTMLResponse)
def page_profil(request: Request, id_adherent: int, db: Session = Depends(get_db)):
    adherent = db.query(Adherent).get(id_adherent)
    emprunts = db.query(Emprunt).filter_by(id_adherent=id_adherent).all()
    return templates.TemplateResponse("profil.html", {"request": request, "adherent": adherent, "emprunts": emprunts})

@app.post("/api/mes-emprunts")
def mes_emprunts(id_adherent: int, db: Session = Depends(get_db)):
    emprunts = db.query(Emprunt).filter_by(id_adherent=id_adherent).all()
    return emprunts

# --- RECOMMANDATION ---
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Charger les modèles entraînés
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
livre_vectors = joblib.load("livre_vectors.pkl")
cosine_sim = joblib.load("cosine_sim_matrix.pkl")

@app.post("/api/recommander-par-description")
def recommander(description: str = Form(...), db: Session = Depends(get_db)):
    # Récupérer tous les livres
    livres = db.query(Livre).all()
    livre_texts = [livre.description if livre.description else livre.titre for livre in livres]

    # Transformer la description saisie
    vect_input = tfidf_vectorizer.transform([description])

    # Similarité avec les livres existants
    similarities = cosine_similarity(vect_input, livre_vectors).flatten()
    indices = np.argsort(similarities)[::-1][:5]

    # Construire la réponse
    suggestions = [
        {
            "id": livres[i].id,
            "titre": livres[i].titre,
            "score": float(similarities[i])
        }
        for i in indices
    ]
    return suggestions

# -------------------- ADMIN --------------------

@app.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard(request: Request, db: Session = Depends(get_db)):
    total_users = db.query(Adherent).count()
    total_books = db.query(Livre).count()
    return templates.TemplateResponse("admin/dashboard.html", {"request": request, "total_users": total_users, "total_books": total_books})

@app.get("/admin/gestion-adherents", response_class=HTMLResponse)
def admin_adherents(request: Request, db: Session = Depends(get_db)):
    adherents = db.query(Adherent).all()
    return templates.TemplateResponse("admin/gestion-adherents.html", {"request": request, "adherents": adherents})

@app.post("/admin/adherents/delete/{id}")
def admin_delete_adherent(id: int, db: Session = Depends(get_db)):
    user = db.query(Adherent).get(id)
    if user:
        db.delete(user)
        db.commit()
    return RedirectResponse("/admin/gestion-adherents", status_code=303)

@app.get("/admin/gestion-livres", response_class=HTMLResponse)
def admin_livres(request: Request, db: Session = Depends(get_db)):
    livres = db.query(Livre).all()
    return templates.TemplateResponse("admin/gestion-livres.html", {"request": request, "livres": livres})

@app.post("/admin/livres/add")
def admin_add_livre(titre: str = Form(...), description: str = Form(...), prix: float = Form(...), image: str = Form(...), disponibilite: int = Form(...), note: int = Form(...), db: Session = Depends(get_db)):
    livre = Livre(titre=titre, description=description, prix=prix, image=image, disponibilite=disponibilite, note=note)
    db.add(livre)
    db.commit()
    return RedirectResponse("/admin/gestion-livres", status_code=303)
@app.get("/admin/emprunts", response_class=HTMLResponse)
def admin_emprunts(request: Request, db: Session = Depends(get_db)):
    emprunts = db.query(Emprunt).all()
    return templates.TemplateResponse(
        "admin/gestion-emprunts.html",
        {"request": request, "emprunts": emprunts}
    )

@app.post("/emprunts")
def add_emprunt(id_adherent: int = Form(...), id_livre: int = Form(...), date_retour_prevue: str = Form(...), db: Session = Depends(get_db)):
    emprunt = Emprunt(
        id_adherent=id_adherent,
        id_livre=id_livre,
        date_retour_prevue=datetime.strptime(date_retour_prevue, "%Y-%m-%d")
    )
    db.add(emprunt)
    livre = db.query(Livre).get(id_livre)
    if livre and livre.disponibilite > 0:
        livre.disponibilite -= 1
    db.commit()
    return RedirectResponse("/admin/emprunts", status_code=303)

@app.post("/retours")
def add_retour(id_emprunt: int = Form(...), db: Session = Depends(get_db)):
    emprunt = db.query(Emprunt).get(id_emprunt)
    if emprunt:
        emprunt.date_retour_effectif = datetime.utcnow()
        emprunt.statut = "retourné"
        livre = db.query(Livre).get(emprunt.id_livre)
        if livre:
            livre.disponibilite += 1
        db.commit()
    return RedirectResponse("/admin/emprunts", status_code=303)

# --- Statistiques ---
@app.get("/admin/statistiques")
def statistiques(db: Session = Depends(get_db)):
    # Top 5 livres empruntés (avec le vrai nombre d’emprunts)
    top_livres = (
        db.query(Livre.id, Livre.titre, func.count(Emprunt.id).label("nb_emprunts"))
        .join(Emprunt)
        .group_by(Livre.id)
        .order_by(func.count(Emprunt.id).desc())
        .limit(5)
        .all()
    )

    # Taux de disponibilité global
    total_livres = db.query(Livre).count()
    livres_disponibles = db.query(Livre).filter(Livre.disponibilite > 0).count()
    taux_disponibilite = int((livres_disponibles / total_livres) * 100) if total_livres else 0

    # Nombre de retards
    nb_retards = db.query(Emprunt).filter(
        Emprunt.statut == "en_cours",
        Emprunt.date_retour_prevue < datetime.utcnow()
    ).count()

    return {
        "top_livres": [
            {"id": l.id, "titre": l.titre, "nb_emprunts": l.nb_emprunts} for l in top_livres
        ],
        "taux_disponibilite": taux_disponibilite,
        "nb_retards": nb_retards
    }
# --- Mes emprunts (corrigé pour JSON) ---
@app.get("/api/mes-emprunts")
def mes_emprunts(id_adherent: int, db: Session = Depends(get_db)):
    emprunts = db.query(Emprunt).filter_by(id_adherent=id_adherent).all()
    return [
        {
            "id": e.id,
            "livre": db.query(Livre).get(e.id_livre).titre if e.id_livre else None,
            "date_emprunt": e.date_emprunt,
            "date_retour_prevue": e.date_retour_prevue,
            "statut": e.statut
        }
        for e in emprunts
    ]