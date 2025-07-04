from audiocraft.models import MusicGen
import torchaudio
import torch
import google.generativeai as genai
import re
import json



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


def extract_as_dict(text):
    # On essaie JSON d'abord
    try:
        return json.loads(text)
    except Exception:
        pass
    # Sinon, on tente format Python dict
    try:
        return ast.literal_eval(text)
    except Exception:
        pass
    # Sinon, on essaie d'extraire un bloc {...}
    matches = re.findall(r'\{[\s\S]*\}', text)
    if matches:
        raw = matches[0]
        try:
            return json.loads(raw)
        except Exception:
            pass
        try:
            return ast.literal_eval(raw)
        except Exception:
            pass
    # Si tout échoue, retourne le texte brut
    return text

def prompt_to_config(prompt, model_name="gemini-1.5-pro-latest"):
    # Exemple d'utilisation
    prompt2 = """
    Context:
    The project "midi-model" (https://github.com/SkyTNT/midi-model) is a toolkit for transformer-based training on MIDI files. Remove model_path, output_midi_path, add time.
    Task:
    Generate a JSON object (not markdown, not explanation), only raw JSON, for midi generation, with instruments, bpm, time, and all other necessary fields.
    """
    prompt = prompt2 + prompt
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        config = extract_as_dict(response.text)
        return config
    except Exception as e:
        print(f"Erreur lors du prompt-to-config Gemini : {e}")
        return None

