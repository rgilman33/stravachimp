from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'), 
    url(r'^authorization/$', views.authorization, name='authorization'),
    url(r'^dashboard/$', views.dashboard, name='dashboard'),
    url(r'^(?P<pk>[0-9]+)/$', views.run_detail, name='run_detail'),
    #url(r'^run_detail/$', views.run_detail, name='run_detail'),
]
