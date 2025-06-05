from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils.translation import gettext_lazy as _


class CustomUser(AbstractUser):
    email = models.EmailField(unique=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username', 'first_name', 'last_name']


class ContactMessage(models.Model):
    email = models.CharField(_("email"), max_length=50)
    message = models.TextField(_("message"))
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Message de {self.email} envoyé à {self.created_at}'
