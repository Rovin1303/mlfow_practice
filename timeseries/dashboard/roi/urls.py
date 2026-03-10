from django.urls import path
from . import views

urlpatterns = [
    path('', views.roi_index, name='roi_index'),
]
