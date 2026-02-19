from django.db import models
from django.contrib.auth.models import User


class Document(models.Model):
    file = models.FileField(upload_to="documents/")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    processed = models.BooleanField(default=False)

    def __str__(self):
        return self.file.name

class ChatSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    document = models.ForeignKey("Document", on_delete=models.CASCADE)
    title = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} ({self.user.username})"

class ChatMessage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        null=True,       # ✅ IMPORTANT
        blank=True
    )
    question = models.TextField()
    answer = models.JSONField()

    # Feedback
    rating = models.IntegerField(null=True, blank=True) 
    rated_at = models.DateTimeField(null=True, blank=True)

    # RL Rating
    rl_state = models.IntegerField(null=True, blank=True)
    rl_action = models.IntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username}: {self.question[:30]}"