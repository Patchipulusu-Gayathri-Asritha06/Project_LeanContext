from django.contrib import admin
from .models import ChatMessage, ChatSession, Document

# Register your models here.
admin.site.register(Document)
admin.site.register(ChatMessage)
admin.site.register(ChatSession)
