# 📊 Introduction à Chart.js

## 📖 Qu’est-ce que Chart.js ?

**Chart.js** est une bibliothèque JavaScript open source qui permet de créer facilement des graphiques interactifs et responsives à partir de données. Elle est simple à utiliser, légère et prend en charge différents types de graphiques.

---

## 🎯 Pourquoi utiliser Chart.js ?

* 📈 **Visualisation claire** des données.
* 🎨 **Design moderne** et personnalisable.
* 📱 **Responsive** (s’adapte aux écrans mobiles et PC).
* 🧩 **Facile à intégrer** dans une page HTML.

---

## ⚙️ Comment ça marche ?

1. Inclure **Chart.js** dans son projet (via CDN ou installation npm).
2. Ajouter un **élément `<canvas>`** dans le HTML pour afficher le graphique.
3. Configurer un objet **JavaScript** contenant :

   * 🔹 Les **données** (labels + valeurs)
   * 🔹 Le **type** de graphique
   * 🔹 Les **options** de personnalisation

---

## 🧰 Types de graphiques disponibles

* 📊 **Barres**
* 📈 **Lignes**
* 🍩 **Doughnut** et 🥧 **Pie (camembert)**
* 📉 **Radar**
* 📦 **Polar Area**
* 🔵 **Scatter (nuage de points)**
* 🔗 **Bubble**

---

## 🔎 Exemple simple — Graphique en barres

```html
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Exemple Chart.js</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <h2>Ventes mensuelles</h2>
  <canvas id="myChart" width="400" height="200"></canvas>

  <script>
    const ctx = document.getElementById('myChart');

    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Jan', 'Fév', 'Mar', 'Avr', 'Mai'],
        datasets: [{
          label: 'Ventes (en €)',
          data: [1200, 1900, 3000, 2500, 3200],
          backgroundColor: 'rgba(54, 162, 235, 0.5)',
          borderColor: 'rgba(54, 162, 235, 1)',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: { beginAtZero: true }
        }
      }
    });
  </script>
</body>
</html>
```

👉 Ce code affiche un **graphique en barres** représentant des ventes mensuelles.

---

## 🔎 Exemple simple — Graphique en lignes

```html
<canvas id="lineChart"></canvas>
<script>
  const ctxLine = document.getElementById('lineChart');
  new Chart(ctxLine, {
    type: 'line',
    data: {
      labels: ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven'],
      datasets: [{
        label: 'Visiteurs',
        data: [50, 75, 60, 90, 100],
        borderColor: 'rgb(255, 99, 132)',
        fill: false
      }]
    }
  });
</script>
```

👉 Ce code trace une **courbe de fréquentation journalière**.

---

## ⚖️ Avantages

* ✅ Simple à apprendre.
* ✅ Compatible avec de nombreux frameworks (React, Vue, Angular).
* ✅ Graphiques interactifs par défaut.

## ⚠️ Limites

* ❌ Moins flexible que **D3.js** pour des visualisations très complexes.
* ❌ Peut être limité en performance avec des **milliers de points**.

---

## 📚 Conclusion

Chart.js est une excellente bibliothèque pour les **débutants** comme pour les projets rapides de **visualisation de données**. Elle permet de créer rapidement des graphiques interactifs et modernes, sans avoir besoin de connaissances avancées en JavaScript.

