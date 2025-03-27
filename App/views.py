from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login as auth_login
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login as auth_login

# Create your views here.
def index(request):
    return render(request, 'App/index.html')

def login_view(request):
    if request.method == "POST":
        username = request.POST.get("my_email")
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
        password = request.POST.get("my_password")
        User.objects.create_user(username=username, password=password)
        return redirect('login')
    return render(request, 'App/register.html')


@login_required
def success_view(request):
    return render(request, 'App/success.html')