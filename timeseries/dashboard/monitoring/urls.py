from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_overview, name='monitoring_index'),
    path('monitoring/', views.drift_monitoring, name='drift_monitoring'),
]
