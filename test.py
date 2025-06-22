from audiocraft.models import MusicGen
from audiocraft.utils.notebook import display_audio

# Charger le modèle (version 'melody' si tu veux inclure une piste)
model = MusicGen.get_pretrained('facebook/musicgen-small')  # ou medium / large

# Donner un prompt texte
model.set_generation_params(duration=10)  # en secondes
descriptions = ["a chill beat with jazzy saxophone"]

# Générer
wav = model.generate(descriptions)

# Jouer ou sauvegarder le résultat
display_audio(wav[0], sample_rate=32000)
