FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /code

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

# 📦 Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    git \
    build-essential \
    ffmpeg \
    sox \
    libsndfile1 \
    libgl1 \
    ninja-build \
    libfluidsynth2 \
    libfluidsynth-dev \
    libffi7 \
    && rm -rf /var/lib/apt/lists/*

# 🔗 Patch si nécessaire (libffi.so.7)
RUN if [ ! -e /usr/lib/x86_64-linux-gnu/libffi.so.7 ]; then \
    ln -s /usr/lib/x86_64-linux-gnu/libffi.so.8 /usr/lib/x86_64-linux-gnu/libffi.so.7 || true; fi

# 📜 Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip uninstall -y cffi pyglet && \
    pip install --no-cache-dir --force-reinstall "cffi==1.15.1" "pyglet==1.5.27"

# 📁 Copier le projet
COPY . .

EXPOSE 8000

# 🚀 Lancer le projet
CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]
