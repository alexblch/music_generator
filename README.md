# Promptune - AI Music Generator

Promptune est une application web alimentée par l'IA qui génère de la musique à partir de prompts textuels. Elle utilise un modèle de deep learning custom entraîné from scratch pour créer des compositions originales au format .wav.

## Fonctionnalités

- Génération de musique à partir d'un prompt textuel
- Interface web en Django
- Modèle IA custom basé sur un Transformer
- Visualisation et téléchargement des musiques générées
- Feedback utilisateur (système de notation)
- Déployé avec Gunicorn + Nginx sur serveur Ubuntu
- Déploiement automatique avec GitHub Actions

## Aperçu

![Screenshot](./media/interface_promptune_demo.png)

## Architecture

User ──> Django ──> IA Generator (Transformer) ──> .wav
                │
                └─> Feedback / Star rating

## Stack Technique

- Python 3.10
- Django 5.1
- PyTorch
- Gunicorn + Nginx
- Conda (env: monenv310)
- GitHub Actions (CI/CD)
- GCP / VM Linux

## Installation locale

```bash
git clone https://github.com/<ton_username>/music_generator.git
cd music_generator
conda create -n monenv310 python=3.10
conda activate monenv310
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

## Déploiement production (Gunicorn + Nginx)

```bash
sudo systemctl start gunicorn
sudo systemctl enable gunicorn
sudo journalctl -fu gunicorn
sudo systemctl restart nginx
```


## CI/CD avec GitHub Actions

Un push sur main déclenche automatiquement :
1. Le build et les tests
2. Une connexion SSH au serveur distant
3. git pull + redémarrage de Gunicorn et Nginx

Fichier .github/workflows/deploy.yml :

```yaml
name: Deploy Django App

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Collect static files
        run: |
          python manage.py collectstatic --noinput

      - name: Deploy via SSH
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.SERVER_HOST }}
          username: ${{ secrets.SERVER_USER }}
          key: ${{ secrets.SERVER_KEY }}
          script: |
            cd ~/music_generator
            git pull origin main
            source /opt/conda/etc/profile.d/conda.sh
            conda activate monenv310
            python manage.py collectstatic --noinput
            sudo systemctl restart gunicorn
            sudo systemctl restart nginx
```


## Licence

PROMPTUNE © 2025 Alexandre BLOCH, Victor JOUIN, Yacine MEKIDECHE
