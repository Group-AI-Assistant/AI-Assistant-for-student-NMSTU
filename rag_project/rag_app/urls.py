# rag_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('ask/', views.ask, name='ask'),
    path('upload/', views.upload_file, name='upload_file'),
    path('upload-page/', views.upload_page, name='upload_page'),  # страница с формой
]