from django.db import models


class ChatHistory(models.Model):
    question   = models.TextField()
    answer     = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
