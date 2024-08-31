from django.urls import path
from .views import analyze_text

url_patterns = [
    path('analyze/', analyze_text, name='analyze_text')
]