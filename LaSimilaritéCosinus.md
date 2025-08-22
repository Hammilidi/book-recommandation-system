# 🔍 La Similarité Cosinus

## 📖 Définition

La **similarité cosinus** est une mesure utilisée pour évaluer à quel point deux vecteurs sont similaires en fonction de l’angle entre eux. Elle est largement utilisée en **traitement du langage naturel (NLP)**, en **systèmes de recommandation** et en **clustering de données**.

➡️ Plus l’angle entre deux vecteurs est petit, plus leur similarité cosinus est élevée (proche de 1).

---

## 🧮 Formule mathématique

Pour deux vecteurs **A** et **B** :

$\text{similarité cosinus} = \cos(\theta) = \frac{A \cdot B}{\|A\| \times \|B\|}$

* 🔹 $A \cdot B$ = produit scalaire des deux vecteurs.
* 🔹 $\|A\|, \|B\|$ = normes (longueurs) des vecteurs.

Valeur :

* **1** → vecteurs identiques (même direction).
* **0** → vecteurs orthogonaux (aucune similarité).
* **-1** → vecteurs opposés (rare en NLP car données non signées).

---

## 🎯 Pourquoi l’utiliser ?

* 📝 **NLP** : comparer des documents/textes.
* 🎵 **Recommandation** : trouver des utilisateurs ou items proches.
* 🧩 **Clustering** : mesurer la ressemblance entre objets.
* 📊 **Réduction de dimension** : travailler dans des espaces vectoriels.

---

## 🔎 Exemple simple (vecteurs numériques)

```python
from numpy import dot
from numpy.linalg import norm

A = [1, 2, 3]
B = [2, 4, 6]

cos_sim = dot(A, B) / (norm(A) * norm(B))
print("Similarité cosinus:", cos_sim)
```

👉 Ici, B est un multiple de A → similarité = **1** (identiques en direction).

---

## 🔎 Exemple simple (textes)

On transforme les textes en vecteurs à l’aide du **TF-IDF**.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Deux phrases
textes = [
    "Le chat mange une souris",
    "Un chat chasse une souris"
]

# Transformation TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(textes)

# Similarité cosinus
sim = cosine_similarity(X[0], X[1])
print("Similarité cosinus:", sim[0][0])
```

👉 Résultat proche de **1**, car les phrases sont très similaires.

---

## 📈 Interprétation

* **≈ 1** → textes/documents très proches.
* **≈ 0.5** → similarité partielle.
* **≈ 0** → aucun lien.

---

## ⚖️ Avantages

* ✅ Simple à calculer.
* ✅ Indépendant de la taille des documents (longueur des vecteurs).
* ✅ Très utilisé en recherche d’information (IR).

## ⚠️ Limites

* ❌ Ne prend pas en compte la **sémantique** (se base uniquement sur les mots exacts).
* ❌ Sensible au **choix de la représentation** (Bag of Words, TF-IDF, embeddings).

---

## 📚 Conclusion

La **similarité cosinus** est un outil fondamental en analyse de données textuelles et vectorielles. Elle permet de mesurer rapidement la proximité entre objets représentés dans un espace vectoriel. Bien qu’elle soit simple et efficace, elle doit souvent être combinée avec des techniques plus avancées (ex. **Word Embeddings, BERT**) pour capturer la vraie sémantique des données.
