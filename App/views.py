from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from App.forms import ContactForm
from App.models import ContactMessage as Contact, MusicGenerated, MidiSentByUsers, FeedBackMusic
from App.utils import generate_music_from_prompt_batch, send_email, prompt_to_config
from django.core.files.storage import default_storage
from django.core.files.storage import FileSystemStorage

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import Project.settings as settings
import os
import random
from dotenv import load_dotenv
import google.generativeai as genai
from io import StringIO
from google.cloud import storage


User = get_user_model()


fs = FileSystemStorage(location=settings.MEDIA_ROOT / 'midis')



client = storage.Client()
bucket = client.bucket('pastorageesgi')
blob = bucket.blob('env_vars/.env')

# Lire le contenu du .env en mémoire
env_data = blob.download_as_text()

# Charger les variables depuis ce contenu (sans fichier)
load_dotenv(stream=StringIO(env_data))


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
    genai.configure(api_key=os.getenv("VERTEX_API_KEY"))
    context = {}

    if request.method == "POST":
        form_type = request.POST.get("form_type")

        if form_type == "generation":
            prompt = request.POST.get("prompt")
            if not prompt:
                context["error"] = "Le prompt ne peut pas être vide."
                return render(request, 'App/generate.html', context)

            config = prompt_to_config(prompt)
            if not config:
                context["error"] = "Erreur dans la génération des paramètres (Gemini)."
                return render(request, 'App/generate.html', context)

            print("🎯 Config Gemini:", config)

            try:
                music_group_id, urls, filenames = generate_music_from_prompt_batch(str(config), n=4)
            except Exception as e:
                context["error"] = f"Erreur lors de la génération de musique : {str(e)}"
                return render(request, 'App/generate.html', context)

            # Sauvegarde des musiques générées
            for fname in filenames:
                MusicGenerated.objects.create(
                    prompt=prompt,
                    music_group_id=music_group_id,
                    generation_config=config,
                    filename=fname
                )

            context.update({
                "generated_musics": urls,
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
