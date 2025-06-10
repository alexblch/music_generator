from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from App.forms import ContactForm
from App.models import ContactMessage as Contact, MusicGenerated, MidiSentByUsers, FeedBackMusic
from App.utils import generate_music_from_prompt
from django.core.files.storage import default_storage
from django.core.files.storage import FileSystemStorage

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import Project.settings as settings
import os




User = get_user_model()


fs = FileSystemStorage(location=settings.MEDIA_ROOT / 'midis')



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
    context = {}

    if request.method == "POST":
        if "prompt" in request.POST:
            # Cas : génération de musique
            prompt = request.POST.get("prompt")
            generated_music_url = generate_music_from_prompt(prompt)
            print("Génération de musique en cours")
            context["generated_music_url"] = generated_music_url

        elif "rating" in request.POST and "feedback" in request.POST:
            # Cas : envoi de feedback
            rating = request.POST.get("rating")
            feedback = request.POST.get("feedback")

            # 💾 Tu peux ici enregistrer les feedbacks en base si besoin
            print(f"Feedback reçu : note={rating}, commentaire={feedback}")
            context["feedback_message"] = "Merci pour votre retour !"

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
            contact.save()
            return redirect('success')
    else:
        form = ContactForm()
    return render(request, 'App/contact.html', {'form': form})

@login_required
def success_view(request):
    return render(request, 'App/success.html')
