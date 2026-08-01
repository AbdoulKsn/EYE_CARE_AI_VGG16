# EyeCare AI — Déploiement Streamlit

## Structure du projet

```
eyecare_app/
├── app.py                     # Application Streamlit (4 pages : Diagnostic, Performance, Historique, À propos)
├── requirements.txt           # Dépendances Python
├── .streamlit/config.toml     # Thème visuel personnalisé (couleurs, police)
├── best_vgg16.keras           # Ton modèle entraîné (à ajouter toi-même)
└── README.md
```

## Fonctionnalités incluses

- **Page Diagnostic** : upload d'image, prédiction avec jauge de confiance, graphique des
  probabilités par classe, carte de résultat colorée par pathologie, alerte spécifique
  glaucome, export du rapport en .txt.
- **Page Performance du modèle** : métriques par classe (précision/rappel/f1), matrice de
  confusion interactive, tableau détaillé.
- **Page Historique** : suivi des analyses de la session, export CSV, graphique de
  répartition des diagnostics.
- **Page À propos** : méthodologie, données, limites, avertissement médical.
- Thème sombre personnalisé (`.streamlit/config.toml`), CSS custom pour un rendu soigné.

## 1. Préparer le dossier en local

1. Crée un dossier `eyecare_app` (ou utilise celui-ci).
2. Copie ton fichier modèle entraîné (`EyeCare_AI_VGG16_Final.keras`, celui à 78,2 %) directement dans ce dossier, à côté de `app.py`.
3. Vérifie que le nom du fichier dans `app.py` (`MODEL_PATH = "EyeCare_AI_VGG16_Final.keras"`) correspond exactement au nom de ton fichier.

## 2. Tester en local avant de déployer

```bash
pip install -r requirements.txt
streamlit run app.py
```

Ça ouvre l'application dans ton navigateur sur `http://localhost:8501`. Teste avec quelques images du dataset pour vérifier que tout fonctionne.

## 3. Déployer sur Streamlit Community Cloud (gratuit)

### Étape A — Mettre le projet sur GitHub
1. Crée un nouveau dépôt GitHub (public ou privé).
2. Pousse-y tout le dossier : `app.py`, `requirements.txt`, `.streamlit/config.toml`, `EyeCare_AI_VGG16_Final.keras`.

⚠️ **Attention à la taille du fichier modèle** : GitHub limite les fichiers à 100 Mo via un push classique.
- Si `EyeCare_AI_VGG16_Final.keras` dépasse cette taille, utilise **Git LFS** :
  ```bash
  git lfs install
  git lfs track "*.keras"
  git add .gitattributes
  git add best_vgg16.keras
  git commit -m "Ajout du modèle avec LFS"
  git push
  ```
- Alternative plus simple : héberger le modèle ailleurs (Google Drive, Hugging Face Hub) et le télécharger au démarrage de l'app (voir section 5 ci-dessous).

### Étape B — Déployer
1. Va sur [share.streamlit.io](https://share.streamlit.io) et connecte-toi avec ton compte GitHub.
2. Clique sur **"New app"**.
3. Sélectionne ton dépôt, la branche (`main`), et le fichier principal (`app.py`).
4. Clique sur **Deploy**.

Le déploiement prend quelques minutes (installation des dépendances, notamment TensorFlow). Une URL publique du type `https://ton-app.streamlit.app` est générée automatiquement.

## 4. Mettre à jour l'application

Chaque `git push` sur la branche connectée redéploie automatiquement l'application.

## 5. Alternative si le modèle est trop volumineux pour GitHub

Héberge `EyeCare_AI_VGG16_Final.keras` sur Hugging Face Hub (gratuit, pas de limite de taille comme GitHub) et télécharge-le au démarrage :

```python
import os
import urllib.request

MODEL_PATH = "EyeCare_AI_VGG16_Final.keras"
MODEL_URL = "https://huggingface.co/<ton-compte>/<ton-repo>/resolve/main/EyeCare_AI_VGG16_Final.keras"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
```

Place ce bloc juste avant `charger_modele()` dans `app.py`.

## 6. Limites à mentionner dans l'app (déjà intégrées)

L'application affiche déjà :
- Un avertissement médical clair (l'outil ne remplace pas un diagnostic).
- Une alerte spécifique quand la classe prédite est "diabetic_retinopathy", classe pour laquelle le modèle a un recall plus faible (~68 %).