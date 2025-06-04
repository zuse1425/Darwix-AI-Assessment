# suggestions/serializers.py

from rest_framework import serializers

class BlogContentSerializer(serializers.Serializer):
    content = serializers.CharField(
        help_text="Entire blog content as a single string",
        allow_blank=False
    )
