from django.urls import path 

from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('start/',views.startsystem,name='start')
    #path('video_feed/', views.video_feed, name='video_feed'),
]
