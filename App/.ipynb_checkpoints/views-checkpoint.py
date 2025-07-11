from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from App.forms import ContactForm
from App.models import ContactMessage as Contact, MusicGenerated, MidiSentByUsers, FeedBackMusic
from App.utils import send_email, prompt_to_config,  get_params_from_prompt, download_from_gcs, run_generation, download_best_model_gcs, synthesize_audios_django, upload_to_gcs, extract_index, lire_fichier_gcs, get_accuracy_from_gcs, download_config_gcs
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
from datetime import datetime
import uuid

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

    bucket_name = "pastorageesgi"
    path_countdown = "countdown.txt"
    countdown = -1

    try:
        countdown = int(lire_fichier_gcs(bucket_name, path_countdown))
    except Exception as e:
        print(f"Erreur lecture countdown.txt sur GCS : {e}")

    context = {"countdown": countdown}

    path_accuracy = "models/bestmodel/stat.txt"
    try:
        accuracy = get_accuracy_from_gcs(bucket_name, path_accuracy)
        context.update({"accuracy": accuracy})
    except Exception as e:
        print(f"Erreur lecture accuracy sur GCS : {e}")
    if request.method == "POST":
        form_type = request.POST.get("form_type")

        # Récupération et clamp de la durée
        try:
            duration_s = int(request.POST.get("duration", 30))
            duration_s = max(10, min(duration_s, 180))
        except (TypeError, ValueError):
            duration_s = 30

        if form_type == "generation":
            # 1) Préparation du device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")

            # 2) Prompt
            prompt = request.POST.get("prompt", "").strip()
            if not prompt:
                context["error"] = "Le prompt ne peut pas être vide."
                return render(request, 'App/generate.html', context)

            # 3) Extraction des params via GPT
            params = get_params_from_prompt(client, prompt)
            if not params:
                context["error"] = "Erreur dans la génération des paramètres (Gemini)."
                return render(request, 'App/generate.html', context)
            params['duration_s'] = duration_s
            print(params['duration_s'])
            
            
            ckpt_path = "/home/jupyter/music_generator/App/tmp/model.safetensors"  # modèle déjà présent
            cfg_path  = download_config_gcs()  # on télécharge seulement la config

            cfg   = MIDIModelConfig.from_json_file(cfg_path)
            model = MIDIModel(cfg)

            torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device_str   = 'cuda:0' if torch.cuda.is_available() else 'cpu'

            # Charger les poids avec safetensors
            state = safe_load_file(ckpt_path, device=device_str)
            model.load_state_dict(state, strict=False)

            # Suite
            model.to(torch_device)
            model.eval()
            tokenizer = model.tokenizer

        
            
            # 4) Télécharger et charger le modèle
            """tmpdir = 'App/tmp/midi_model'
            os.makedirs(tmpdir, exist_ok=True)
            ckpt_path = download_best_model_gcs()
            download_from_gcs(bucket_name, CONFIG_BLOB, f"{tmpdir}/config.json")
            
            cfg   = MIDIModelConfig.from_json_file(f"{tmpdir}/config.json")
            model = MIDIModel(cfg)
            state = safe_load_file(ckpt_path)
            model.load_state_dict(state, strict=False)"""
            
            
            """model.to(device)
            model.eval()
            tokenizer = model.tokenizer"""
            
            # 5) Préparer le synthétiseur + thread pool
            sf    = hf_hub_download('skytnt/midi-model','soundfont.sf2')
            synth = MidiSynthesizer(sf)
            pool  = ThreadPoolExecutor(max_workers=BATCH_SIZE)

            # 6) Génération des séquences MIDI
            try:
                seqs = run_generation(
                    model, tokenizer, synth, pool,
                    params['instruments'], params['drum_kit'], params['bpm'],
                    params['time_signature'], 
                    KEY_SIG_LIST.index(params['key_signature']),
                    params['seed'], params['random_seed'],
                    params['duration_s'] * EVENTS_PER_SECOND,
                    params['temperature'], params['top_p'], params['top_k'],
                    params['allow_cc']
                )
            except Exception as e:
                context["error"] = f"Erreur lors de la génération de musique : {e}"
                return render(request, 'App/generate.html', context)

            # 7) Sauvegarde des fichiers .mid
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            midi_paths = []
            for i, seq in enumerate(seqs, start=1):
                chunks     = [
                    seq[j:j+tokenizer.max_token_seq] +
                    [tokenizer.pad_id] * (tokenizer.max_token_seq - len(seq[j:j+tokenizer.max_token_seq]))
                    for j in range(0, len(seq), tokenizer.max_token_seq)
                ]
                midi_score = tokenizer.detokenize(chunks)
                midi_name  = f"out_{i}.mid"
                midi_path  = OUTPUT_DIR / midi_name
                with open(midi_path, 'wb') as f:
                    f.write(score2midi(midi_score))
                midi_paths.append(str(midi_path))

            # 8) Synthèse parallèle des WAV tronçonnés à duration_s
            os.makedirs(AUDIO_DIR, exist_ok=True)
            wav_filenames = synthesize_audios_django(
                midi_paths,
                synth,
                pool,
                duration_s,
                str(AUDIO_DIR)
            )

            # 9) Enregistrer en base
            music_group_id = str(uuid.uuid4())
            for m in midi_paths:
                MusicGenerated.objects.create(
                    prompt=prompt,
                    music_group_id=music_group_id,
                    generation_config=params,
                    filename=os.path.basename(m)
                )
            for w in wav_filenames:
                MusicGenerated.objects.create(
                    prompt=prompt,
                    music_group_id=music_group_id,
                    generation_config=params,
                    filename=w
                )

            # 10) Préparer les URLs pour le template
            mid_urls = [f"{settings.MEDIA_URL}outputs/{os.path.basename(p)}" for p in midi_paths]
            wav_urls = [f"{settings.MEDIA_URL}audio/{w}" for w in wav_filenames]
            context.update({
                "zipped_musics": list(zip(wav_urls, mid_urls)),
                "music_group_id": music_group_id,
                "prompt": prompt
            })

        elif form_type == "feedback":
            music_group_id = request.POST.get("music_group_id")
            try:
                preferred_version = int(request.POST.get("preferred_version"))  # Index dans la liste affichée
                print(f"Version préférée : {preferred_version}")

                # Regénère les mêmes listes que côté génération
                output_dir = os.path.join(settings.MEDIA_ROOT, "outputs")
                midi_paths = sorted(
                    [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".mid")],
                    key=extract_index
                )
                wav_filenames = sorted(
                    [f for f in os.listdir(os.path.join(settings.MEDIA_ROOT, "audio")) if f.endswith(".wav")],
                    key=extract_index
                )

                # Protection : vérifier que l'index est dans la bonne plage
                if preferred_version < 1 or preferred_version > len(midi_paths):
                    raise ValueError(f"Index préféré invalide : {preferred_version}")

                # Récupère le chemin MIDI exact sélectionné par l'utilisateur
                midi_path = midi_paths[preferred_version - 1]
                midi_filename = os.path.basename(midi_path)

                # Enregistrement du feedback
                FeedBackMusic.objects.create(
                    music_group_id=music_group_id,
                    preferred_version=preferred_version
                )

                # Upload vers GCS
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                blob_name = f"data/batch/{midi_filename.replace('.mid','')}_{timestamp}.mid"
                upload_to_gcs(midi_path, bucket_name, blob_name)

                context["feedback_message"] = "Merci pour votre retour !"

            except Exception as e:
                context["feedback_message"] = f"Erreur lors du feedback : {e}"

    return render(request, 'App/generate.html', context)







@csrf_exempt
def upload_midi_ajax(request):
    if request.method == 'POST' and request.FILES.get('midi_file'):
        midi_file = request.FILES['midi_file']
        if not midi_file.name.endswith(('.mid', '.midi')):
            return JsonResponse({'error': 'Format non supporté'}, status=400)

        # Création d'un nom de fichier unique avec date
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        base, ext = os.path.splitext(midi_file.name)
        new_filename = f"{base}-{timestamp}{ext}"

        # Stockage dans media/midis/
        midi_storage = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'midis'))
        filename = midi_storage.save(new_filename, midi_file)

        # URL correcte pour accès immédiat
        url = settings.MEDIA_URL + 'midis/' + filename
        return JsonResponse({'success': True, 'url': url})

    return JsonResponse({'error': 'Aucun fichier'}, status=400)


def help(request):
    audio_url = None

    if request.method == 'POST':
        if 'new_midi' in request.FILES:
            file = request.FILES['new_midi']
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            base, ext = os.path.splitext(file.name)
            new_filename = f"{base}-{timestamp}{ext}"
            path = default_storage.save('midis/' + new_filename, file)
            audio_url = path

        elif 'audio_url' in request.POST:
            audio_url = request.POST['audio_url']
            print("Fichier validé par l'utilisateur :", audio_url)
            midi_sent = MidiSentByUsers(
                midi_file=audio_url,
                user=request.user
            )
            midi_sent.save()
            # TODO : éventuel upload vers GCS ici aussi
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
