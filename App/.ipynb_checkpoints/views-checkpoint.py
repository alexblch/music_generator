from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from App.forms import ContactForm
from App.models import ContactMessage as Contact, MusicGenerated, MidiSentByUsers, FeedBackMusic
from App.utils import send_email, prompt_to_config,  get_params_from_prompt, download_from_gcs, run_generation, download_best_model_gcs
from django.core.files.storage import default_storage
from django.core.files.storage import FileSystemStorage
from concurrent.futures import ThreadPoolExecutor, as_completed
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.http import JsonResponse
import Project.settings as settings
import os
import random
from dotenv import load_dotenv
import google.generativeai as genai
from io import StringIO
from google.cloud import storage
from openai import OpenAI
import torch
import numpy as np 
from django.conf import settings
from App.midi_model.MIDI import score2midi, midi2score, score2opus, Number2patch
from safetensors.torch import load_file as safe_load_file
from huggingface_hub import hf_hub_download
from App.midi_model.midi_model import MIDIModel, MIDIModelConfig
from App.midi_model.midi_tokenizer import MIDITokenizer
from App.midi_model.midi_synthesizer import MidiSynthesizer
from subprocess import run


User = get_user_model()


fs = FileSystemStorage(location=settings.MEDIA_ROOT / 'midis')

# --- Mappings globaux ---
patch2number = {v: k for k, v in Number2patch.items()}
number2drum_kits = {
    -1: "None", 0: "Standard", 8: "Room", 16: "Power",
    24: "Electric", 25: "TR-808", 32: "Jazz", 40: "Blush", 48: "Orchestra"
}
drum_kits2number = {v: k for k, v in number2drum_kits.items()}

client = storage.Client()
bucket = client.bucket('pastorageesgi')
blob = bucket.blob('env_vars/.env')

# Lire le contenu du .env en mémoire
env_data = blob.download_as_text()

# Charger les variables depuis ce contenu (sans fichier)
load_dotenv(stream=StringIO(env_data))

# Constantes
EVENTS_PER_SECOND = 16
BATCH_SIZE       = 4
AUDIO_DIR = settings.MEDIA_ROOT / "audio"
BUCKET           = 'pastorageesgi'
MODEL_BLOB       = 'models/bestmodel/model.safetensors'
CONFIG_BLOB      = 'models/bestmodel/config.json'
OUTPUT_DIR = settings.MEDIA_ROOT / "outputs"

# --- Extraction des params via GPT function call ---
INSTRUMENTS      = list(patch2number.keys())
DRUM_KIT_LIST    = list(drum_kits2number.keys())
TIME_SIG_LIST    = ["auto","2/4","3/4","4/4","5/4","6/8","7/8","9/8","12/8"]
KEY_SIG_LIST     = ["C","G","D","A","E","B","F#","C#","F","Bb","Eb","Ab","Db","Gb","Cb"]

# Create your views here.
def index(request):
    return render(request, 'App/index.html')

def login_view(request):
    if request.method == "POST":
        username = request.POST.get("my_username")
        password = request.POST.get("my_password")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            auth_login(request, user)
            return redirect('success')
        else:
            return render(request, 'App/login.html', {"error": "Identifiants incorrects"})
    return render(request, 'App/login.html')

def register(request):
    if request.method == "POST":
        username = request.POST.get("my_username")
        firstname = request.POST.get("first_name")
        lastname = request.POST.get("last_name")
        email = request.POST.get("my_email")
        password = request.POST.get("my_password")
        if not username or not password:
            return render(request, 'App/register.html', {"error": "Tous les champs sont requis."})
        
        if User.objects.filter(username=username).exists():
            return render(request, 'App/register.html', {
            "error": "Ce nom d’utilisateur est déjà utilisé. Veuillez en choisir un autre."
        })

        user = User.objects.create_user(
            username=username,
            first_name=firstname,
            last_name=lastname,
            email=email,
            password=password
        )
        user.save()
        return redirect('login')
    return render(request, 'App/register.html')

def logout(request):
    auth_logout(request)
    return redirect('index')



def generate_music(request):
    client = OpenAI(api_key=os.getenv('OPEN_API_KEY'))
    context = {}

    if request.method == "POST":
        form_type = request.POST.get("form_type")

        if form_type == "generation":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            prompt = request.POST.get("prompt")
            if not prompt:
                context["error"] = "Le prompt ne peut pas être vide."
                return render(request, 'App/generate.html', context)

            params = get_params_from_prompt(client, prompt)
            if not params:
                context["error"] = "Erreur dans la génération des paramètres (Gemini)."
                return render(request, 'App/generate.html', context)

            print("🎯 Config OpenAI:", params)
            
            # Download model & config to temporary directory
            tmpdir = 'App/midi_model/tmp/model'
            path = download_best_model_gcs()
            os.makedirs(tmpdir, exist_ok=True)
            download_from_gcs(BUCKET, CONFIG_BLOB, f"{tmpdir}/config.json")

            # Load model and move to device
            cfg = MIDIModelConfig.from_json_file(f"{tmpdir}/config.json")
            model = MIDIModel(cfg)
            state = safe_load_file(path)
            model.load_state_dict(state, strict=False)
            model.to(device)
            model.eval()
            tokenizer = model.tokenizer
            print("modèles chargés")
            
            # Initialize synthesizer and thread pool
            synth = MidiSynthesizer(hf_hub_download('skytnt/midi-model','soundfont.sf2'))
            pool = ThreadPoolExecutor(max_workers=BATCH_SIZE)
            print("Synth et thread pool importés")

            try:
                # Generate MIDI sequences on GPU
                print("génération en cours")
                seqs = run_generation(
                    model, tokenizer, synth, pool,
                    params['instruments'], params['drum_kit'], params['bpm'],
                    params['time_signature'], KEY_SIG_LIST.index(params['key_signature']),
                    params['seed'], params['random_seed'], params['duration_s']*EVENTS_PER_SECOND,
                    params['temperature'], params['top_p'], params['top_k'], params['allow_cc']
                )
                # Save MIDI files
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                os.makedirs(AUDIO_DIR, exist_ok=True)
            except Exception as e:
                context["error"] = f"Erreur lors de la génération de musique : {str(e)}"
                return render(request, 'App/generate.html', context)

            import uuid

            music_group_id = str(uuid.uuid4())
            mid_filenames = []
            wav_filenames = []

            for i, seq in enumerate(seqs, start=1):
                # -- Chunks + detokenize --
                chunks = [seq[j:j+tokenizer.max_token_seq] +
                          [tokenizer.pad_id] * (tokenizer.max_token_seq - len(seq[j:j+tokenizer.max_token_seq]))
                          for j in range(0, len(seq), tokenizer.max_token_seq)]
                midi_score = tokenizer.detokenize(chunks)

                # -- .mid : OUTPUT_DIR --
                midi_path = OUTPUT_DIR / f"out_{i}.mid"
                with open(midi_path, 'wb') as f:
                    f.write(score2midi(midi_score))
                mid_filenames.append(f"out_{i}.mid")

                # -- .wav : AUDIO_DIR --
                wav_path = AUDIO_DIR / f"out_{i}.wav"
                run(["fluidsynth", "-ni", "soundfont.sf2", str(midi_path), "-F", str(wav_path), "-r", "44100"])
                wav_filenames.append(f"out_{i}.wav")

                # -- Enregistrement base (seulement MIDI ou les deux si tu veux) --
                MusicGenerated.objects.create(
                    prompt=prompt,
                    music_group_id=music_group_id,
                    generation_config=params,
                    filename=f"out_{i}.mid"
                )
                MusicGenerated.objects.create(
                    prompt=prompt,
                    music_group_id=music_group_id,
                    generation_config=params,
                    filename=f"out_{i}.wav"
                )

            # Création des URLs accessibles côté template
            mid_urls = [f"{settings.MEDIA_URL}outputs/{fname}" for fname in mid_filenames]
            wav_urls = [f"{settings.MEDIA_URL}audio/{fname}" for fname in wav_filenames]

            zipped_musics = zip(wav_urls, mid_urls)  # wav en premier pour lecture

            context.update({
                "zipped_musics": list(zipped_musics),
                "music_group_id": music_group_id,
                "prompt": prompt
            })

        elif form_type == "feedback":
            try:
                music_group_id = request.POST.get("music_group_id")
                preferred_version = int(request.POST.get("preferred_version"))

                if music_group_id and 0 <= preferred_version <= 3:
                    FeedBackMusic.objects.create(
                        music_group_id=music_group_id,
                        preferred_version=preferred_version + 1
                    )
                    context["feedback_message"] = "Merci pour votre retour !"
                else:
                    context["feedback_message"] = "Feedback invalide ou incomplet."
            except Exception as e:
                context["feedback_message"] = f"Erreur lors du feedback : {str(e)}"

    return render(request, 'App/generate.html', context)





@csrf_exempt
def upload_midi_ajax(request):
    if request.method == 'POST' and request.FILES.get('midi_file'):
        midi_file = request.FILES['midi_file']
        if not midi_file.name.endswith(('.mid', '.midi')):
            return JsonResponse({'error': 'Format non supporté'}, status=400)

        # Stockage dans media/midis/
        midi_storage = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'midis'))
        filename = midi_storage.save(midi_file.name, midi_file)

        # URL correcte pour accès immédiat
        url = settings.MEDIA_URL + 'midis/' + filename
        return JsonResponse({'success': True, 'url': url})

    return JsonResponse({'error': 'Aucun fichier'}, status=400)


def help(request):
    audio_url = None

    if request.method == 'POST':
        if 'new_midi' in request.FILES:
            # upload classique (non utilisé ici mais conservé)
            file = request.FILES['new_midi']
            path = default_storage.save('midis/' + file.name, file)
            audio_url = path

        elif 'audio_url' in request.POST:
            # cas du formulaire AJAX "Envoyer"
            audio_url = request.POST['audio_url']
            print("Fichier validé par l'utilisateur :", audio_url)
            midi_sent = MidiSentByUsers(
                midi_file=audio_url,
                user=request.user
            )
            # tu peux enregistrer ça en base ici si tu veux
            midi_sent.save()
            #todo: envoyer vers compute storage gcp
            pass
    return render(request, 'App/help.html', {
        'audio': audio_url
    })


def contact(request):
    if request.method == "POST":
        form = ContactForm(request.POST)
        if form.is_valid():
            # Process the form data
            name = form.cleaned_data['name']
            email = form.cleaned_data['email']
            message = form.cleaned_data['message']
            # Send email or save to database
            contact = Contact(name=name, email=email, message=message)
            load_dotenv()
            send_email(
                sender_email=os.getenv('SENDER_EMAIL'),
                sender_password=os.getenv('SENDER_PASSWORD'),
                receiver_email=os.getenv('RECEIVER_EMAIL'),
                subject=f"Contact from {name}",
                body=f'Message from {name}, ({email}):\n\n{message}'
            )
            contact.save()
            return redirect('success')
    else:
        form = ContactForm()
    return render(request, 'App/contact.html', {'form': form})

@login_required
def success_view(request):
    return render(request, 'App/success.html')
