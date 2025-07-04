FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /code

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

# 🛠️ Installer dépendances système
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata \
        git \
        build-essential \
        ffmpeg \
        sox \
        libsndfile1 \
        libgl1 \
        ninja-build \
        && rm -rf /var/lib/apt/lists/*

# 🐍 Installer dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 🧠 Pour forcer la compilation de xFormers (si git install dans requirements)
# Sinon, commenter cette ligne si déjà dans requirements.txt
# RUN pip install git+https://github.com/facebookresearch/xformers.git

# 📁 Copier le projet
COPY . .

EXPOSE 8000

# 🚀 Lancer Django avec migrations
CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]
