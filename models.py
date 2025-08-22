from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from database import Base 

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
    image = Column(String(500))       # correspond au CSV
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
