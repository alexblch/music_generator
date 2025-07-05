from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils.translation import gettext_lazy as _
from django.core.validators import MinValueValidator, MaxValueValidator
from django.conf import settings


class CustomUser(AbstractUser):
    email = models.EmailField(unique=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username', 'first_name', 'last_name']


class ContactMessage(models.Model):
    name = models.CharField(_("name"), max_length=50)
    email = models.CharField(_("email"), max_length=50)
    message = models.TextField(_("message"))
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Message de {self.email} envoyé à {self.created_at}, par {self.name}'


class MusicGenerated(models.Model):
    prompt = models.TextField(_("Prompt de génération"), max_length=500)
    music_group_id = models.CharField(_("ID de groupe musique"), max_length=100)
    generation_config = models.JSONField(_("Paramètres de génération (config Gemini)"))
    filename = models.CharField(_("Nom du fichier généré"), max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"🎵 {self.created_at.strftime('%Y-%m-%d %H:%M')} – {self.prompt[:50]}..."



class MidiSentByUsers(models.Model):
    midi_file = models.FileField(_("midi file"), upload_to='midis/')
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='uploaded_midis')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'MIDI file uploaded by {self.user.username} at {self.created_at}'


class FeedBackMusic(models.Model):
    music_group_id = models.CharField(_("ID du groupe évalué"), max_length=100)
    preferred_version = models.PositiveSmallIntegerField(_("Version préférée (1 à 4)"))
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Feedback sur {self.music_group_id} – version {self.preferred_version}"
