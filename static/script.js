document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
        loginForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData(loginForm);
            const data = new URLSearchParams(formData);
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: data
            });
            const result = await response.json();
            const errorMessage = document.getElementById('error-message');
            if (response.ok) {
                window.location.href = '/home';
            } else {
                errorMessage.textContent = result.detail || 'Erreur de connexion';
            }
        });
    }

    const registerForm = document.getElementById('register-form');
    if (registerForm) {
        registerForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData(registerForm);
            const data = Object.fromEntries(formData.entries());
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();
            const errorMessage = document.getElementById('error-message');
            if (response.ok) {
                alert('Inscription réussie ! Vous pouvez maintenant vous connecter.');
                window.location.href = '/';
            } else {
                errorMessage.textContent = result.detail || 'Erreur d\'inscription';
            }
        });
    }

    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        searchInput.addEventListener('input', async function(e) {
            const query = e.target.value;
            const response = await fetch(`/api/livres?search=${query}`);
            const livres = await response.json();
            const booksList = document.getElementById('books-list');
            booksList.innerHTML = ''; // Clear previous results
            livres.forEach(livre => {
                const livreElement = `
                    <div class="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-xl transition-shadow duration-300">
                        <a href="/livre/${livre.id}">
                            <img src="${livre.image_url}" alt="${livre.titre}" class="w-full h-56 object-cover">
                            <div class="p-4">
                                <h3 class="text-lg font-semibold text-gray-800">${livre.titre}</h3>
                                <p class="text-sm text-gray-500 mt-1">
                                    <span class="text-${livre.disponibilite > 0 ? 'green' : 'red'}-500 font-medium">
                                        ${livre.disponibilite > 0 ? 'Disponible' : 'Réservé'}
                                    </span>
                                </p>
                            </div>
                        </a>
                    </div>
                `;
                booksList.innerHTML += livreElement;
            });
        });
    }

    const reserverBtn = document.getElementById('reserver-btn');
    if (reserverBtn) {
        reserverBtn.addEventListener('click', async function() {
            const livreId = reserverBtn.dataset.livreId;
            const response = await fetch(`/api/reservations/${livreId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            const result = await response.json();
            const reservationMessage = document.getElementById('reservation-message');
            if (response.ok) {
                reservationMessage.innerHTML = `<div class="bg-green-100 text-green-700 p-4 rounded-lg">${result.message}</div>`;
                reserverBtn.disabled = true;
                reserverBtn.textContent = 'Réservé';
            } else {
                reservationMessage.innerHTML = `<div class="bg-red-100 text-red-700 p-4 rounded-lg">${result.detail}</div>`;
            }
        });
    }

    const recoForm = document.getElementById('reco-form');
    if (recoForm) {
        recoForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const description = document.getElementById('description-input').value;
            const recoResults = document.getElementById('reco-results');
            const recoMessage = document.getElementById('reco-message');

            recoResults.innerHTML = '';
            recoMessage.textContent = 'Recherche en cours...';
            recoMessage.classList.remove('hidden');

            const response = await fetch('/api/recommander-par-description', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ description: description })
            });

            const livres = await response.json();
            if (response.ok) {
                recoMessage.classList.add('hidden');
                livres.forEach(livre => {
                    const livreElement = `
                        <div class="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-xl transition-shadow duration-300">
                            <a href="/livre/${livre.id}">
                                <img src="${livre.image_url}" alt="${livre.titre}" class="w-full h-56 object-cover">
                                <div class="p-4">
                                    <h3 class="text-lg font-semibold text-gray-800">${livre.titre}</h3>
                                    <p class="text-sm text-gray-500 mt-1">
                                        <span class="text-${livre.disponibilite > 0 ? 'green' : 'red'}-500 font-medium">
                                            ${livre.disponibilite > 0 ? 'Disponible' : 'Réservé'}
                                        </span>
                                    </p>
                                </div>
                            </a>
                        </div>
                    `;
                    recoResults.innerHTML += livreElement;
                });
            } else {
                recoMessage.textContent = livres.detail || 'Erreur lors de la recommandation';
                recoMessage.classList.remove('hidden');
            }
        });
    }

    const statsPage = document.querySelector('[data-page="stats"]');
    if (statsPage) {
        async function fetchStats() {
            const response = await fetch('/api/statistiques');
            const stats = await response.json();
            const ctx = document.getElementById('myChart').getContext('2d');
            const popularBooksData = stats.livres_plus_empruntes.map(b => b.emprunts);
            const popularBooksLabels = stats.livres_plus_empruntes.map(b => b.titre);

            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: popularBooksLabels,
                    datasets: [{
                        label: 'Nombre d\'emprunts',
                        data: popularBooksData,
                        backgroundColor: 'rgba(59, 130, 246, 0.5)',
                        borderColor: 'rgba(59, 130, 246, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });

            document.getElementById('dispo-rate').textContent = `${stats.taux_disponibilite}%`;
            document.getElementById('late-count').textContent = stats.nombre_retards;
        }
        fetchStats();
    }
});