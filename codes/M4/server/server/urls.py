from django.contrib import admin
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index),
    path("process", views.upload_process)
]+  static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
