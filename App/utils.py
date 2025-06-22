from audiocraft.models import MusicGen
import torchaudio
import torch



import os

def generate_music_from_prompt(prompt):
    """
    Génère une musique à partir d'un prompt et retourne l'URL du fichier audio.
    """
    if not prompt:
        return None
    
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    model.set_generation_params(duration=10)

    # Génération
    generated_music = model.generate([prompt])

    # Définir le nom du fichier (ex: output_123456.wav)
    file_name = f"output_{abs(hash(prompt)) % 1000000}.wav"
    file_path = os.path.join("media", file_name)

    # Crée le dossier s'il n'existe pas
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Sauvegarder dans le dossier media/
    torchaudio.save(file_path, generated_music[0].cpu(), 32000)

    # Retourne l'URL à utiliser dans la balise <audio>
    return f"/media/{file_name}"


# prompt: envoi mail python

import smtplib
from email.mime.text import MIMEText

def send_email(sender_email, sender_password, receiver_email, subject, body):
    """Sends an email using Gmail's SMTP server."""
    try:
        # Create a text/plain message
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email

        # Connect to the server
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_bytes())

        print("Email sent successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

