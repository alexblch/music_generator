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
    prompt = models.TextField(_("prompt"), max_length=500)
    music = models.FileField(_("music"), upload_to='music/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Music generated for prompt: {self.prompt}'


class MidiSentByUsers(models.Model):
    midi_file = models.FileField(_("midi file"), upload_to='midis/')
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='uploaded_midis')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'MIDI file uploaded by {self.user.username} at {self.created_at}'


from django.conf import settings

class FeedBackMusic(models.Model):

    promptfeed = models.TextField(_("prompt"), max_length=500)
    rate = models.IntegerField(_("rate"), validators=[MinValueValidator(1), MaxValueValidator(5)])
    reward = models.FloatField(_("reward"), default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
