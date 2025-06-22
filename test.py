from audiocraft.models import MusicGen
import torchaudio
import torch



model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=10)
descriptions = ["a chill beat with jazzy saxophone"]

wav = model.generate(descriptions)

# Sauvegarder l'audio
torchaudio.save("output.wav", wav[0].cpu(), 32000)
print("✅ Audio généré : output.wav")