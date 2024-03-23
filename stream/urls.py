"""stream URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# 2019-07-23 Modified by Tran Le Anh

from django.contrib import admin
from django.urls import path
from webcam.views import index, video_feed_1, camera_1
# from webcam.views import database, search

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='home'),
    path('video_feed_1/', video_feed_1, name="video-feed-1"),
    path('camera1/', camera_1, name= 'stream'),
    # path('index/database/', database),
    # path('index/database/50latest', database),
    # path('index/database/search', search),
]
