from django.contrib import admin
from django.urls import path, include
from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.login_view, name='login'), 
    path('register/', views.register, name='register'),
    path('success/', views.success_view, name='success'),
    path('logout/', views.logout, name='logout'),
    path('generate/', views.generate_music, name='generate'),
    path('contact/', views.contact, name='contact'),
    path('help/', views.help, name='help'),
    path('upload-midi/', views.upload_midi_ajax, name='upload_midi_ajax'),
]