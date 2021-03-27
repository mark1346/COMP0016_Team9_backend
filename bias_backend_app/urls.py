from django.urls import include, path
from rest_framework import routers
from . import views

# router = routers.DefaultRouter()
# router.register(r'WordPairs', views.WordPairViewSet)
# router.register(r'Word', views.WordViewSet)

urlpatterns = [
    # path('', include(router.urls)),
    # path('api-auth', include('rest_framework.urls', namespace="rest_framework")),
    path('handle_request', views.handleRequest.as_view()),
    #path('handle_substitution_request', views.handleSubstituteRequest.as_view())
]