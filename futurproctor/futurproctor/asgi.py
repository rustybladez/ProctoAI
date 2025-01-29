import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from proctoring.routing import websocket_urlpatterns  # Import WebSocket URL patterns

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'futurproctor.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(  # WebSocket handling
        URLRouter(
            websocket_urlpatterns  # Use routing from the proctoring app
        )
    ),
})
