from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model

User = get_user_model()


# Create your views here.
def index(request):
    return render(request, 'App/index.html')

def login_view(request):
    if request.method == "POST":
        email = request.POST.get("my_email")
        password = request.POST.get("my_password")
        user = authenticate(request, username=email, password=password)
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


@login_required
def success_view(request):
    return render(request, 'App/success.html')