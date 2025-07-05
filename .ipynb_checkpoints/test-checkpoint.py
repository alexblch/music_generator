from io import StringIO
from dotenv import load_dotenv
from google.cloud import storage
import os

client = storage.Client()
bucket = client.bucket('pastorageesgi')
blob = bucket.blob('env_vars/.env')

# Lire le contenu du .env en mémoire
env_data = blob.download_as_text()

# Charger les variables depuis ce contenu (sans fichier)
load_dotenv(stream=StringIO(env_data))

# Exemple
print("VERTEX_API_KEY =", os.getenv("VERTEX_API_KEY"))
