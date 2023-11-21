from django.urls import path

from app import views

app_name = 'app'

urlpatterns = [
    path('', views.index),
    path('detail/<int:result_id>/',views.detail,name="detail")
]
