# 🕸️ Introduction au Web Scraping

## 📖 Qu’est-ce que le Web Scraping ?

Le **Web Scraping** est une technique informatique qui consiste à extraire automatiquement des données disponibles sur des sites web. Au lieu de copier manuellement les informations, un programme (appelé *scraper*) collecte les données de manière automatique et les organise dans un format exploitable (CSV, JSON, base de données, etc.).

---

## 🎯 Pourquoi utiliser le Web Scraping ?

* 📊 **Collecte de données massives** pour l’analyse (marchés, tendances, recherches scientifiques).
* 🏷️ **Veille concurrentielle** (suivi des prix, produits, avis clients).
* 📰 **Agrégation de contenus** (actualités, annonces immobilières, offres d’emploi).
* 🤖 **Alimentation d’IA/ML** avec des données réelles.

---

## ⚙️ Comment ça marche ?

Un processus de web scraping se déroule généralement en 4 étapes :

1. 🌐 **Accéder à la page web** (via une requête HTTP).
2. 🔎 **Lire le code HTML** et analyser la structure (DOM).
3. 🧩 **Extraire les données utiles** (titres, prix, liens, textes…).
4. 💾 **Enregistrer les données** dans un format exploitable (CSV, JSON, BDD).

---

## 🧰 Outils courants

* 🟦 **Requests** : pour télécharger le contenu HTML.
* 🌿 **BeautifulSoup** : pour analyser et extraire les informations du HTML.
* 🕷️ **Scrapy** : framework complet pour gérer des projets de scraping à grande échelle.
* 🎭 **Selenium / Playwright** : pour interagir avec des pages dynamiques (JavaScript).

---

## ⚖️ Règles et bonnes pratiques

* ✅ Vérifier les **conditions d’utilisation** du site et respecter la légalité.
* 🪪 Respecter le fichier **robots.txt** (qui précise ce que le site autorise).
* 🕒 Ne pas surcharger le serveur : ajouter des **pauses** entre les requêtes.
* 📩 Utiliser les **APIs officielles** si elles existent.

---

## 🔎 Exemple simple (statique)

```python
import requests
from bs4 import BeautifulSoup

url = "https://books.toscrape.com/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

for book in soup.select("article.product_pod h3 a"):
    print(book["title"])
```

👉 Ici, on récupère simplement la liste des **titres de livres** affichés sur la page d’accueil.

---

## 🔎 Exemple simple (dynamique)

```python
from selenium import webdriver
from selenium.webdriver.common.by import By

browser = webdriver.Chrome()
browser.get("https://quotes.toscrape.com/js/")

quotes = browser.find_elements(By.CLASS_NAME, "text")
for q in quotes:
    print(q.text)

browser.quit()
```

👉 Ici, on utilise **Selenium** pour récupérer des citations générées par JavaScript.

---

## 📚 Conclusion

Le web scraping est une méthode puissante pour collecter et analyser des données issues du web. Avec des outils simples comme **Requests + BeautifulSoup**, ou plus avancés comme **Scrapy** et **Selenium**, il est possible d’automatiser la collecte d’informations pour la recherche, la veille et l’analyse de données.

⚠️ Mais attention : il doit toujours être pratiqué de façon **responsable, légale et éthique**.
