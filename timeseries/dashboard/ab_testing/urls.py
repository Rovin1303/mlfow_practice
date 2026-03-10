from django.urls import path
from . import views

urlpatterns = [
    path('', views.ab_testing_index, name='ab_testing_index'),
]
