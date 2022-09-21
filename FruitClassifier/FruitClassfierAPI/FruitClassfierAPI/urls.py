from django.contrib import admin
from django.urls import path
from fruitclassifier.views import FruitClassifierView


urlpatterns = [
    #path('admin/', admin.site.urls),
    path('',FruitClassifierView)
]
