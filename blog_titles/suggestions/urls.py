# suggestions/urls.py

from django.urls import path
from .views import TitleSuggestionAPIView

urlpatterns = [
    path("suggest-titles", TitleSuggestionAPIView.as_view(), name="suggest-titles"),
]
