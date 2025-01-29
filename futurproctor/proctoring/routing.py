from django.urls import path
from . import consumers

websocket_urlpatterns = [
    path('ws/exam_detections/', consumers.ExamDetectionConsumer.as_asgi()),  # WebSocket URL
]
