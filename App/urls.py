from django.contrib import admin
from django.urls import path, include
from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.login_view, name='login'), 
    path('register/', views.register, name='register'),
    path('success/', views.success_view, name='success'),
    path('logout/', views.logout, name='logout')
]