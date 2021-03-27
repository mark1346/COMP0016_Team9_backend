from django.contrib import admin
from .models import WordPair, Word, UserConfig

# Register your models here.

admin.site.register(WordPair)
admin.site.register(Word)
admin.site.register(UserConfig)
