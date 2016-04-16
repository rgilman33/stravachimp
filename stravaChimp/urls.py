from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'), 
    url(r'^authorization/$', views.authorization, name='authorization'),
    url(r'^(?P<athleteId>[0-9]+)/dashboard/$', views.dashboard, name='dashboard'),
    url(r'^(?P<athleteId>[0-9]+)/run_detail/(?P<activityId>[0-9]+)/$', views.run_detail, name='run_detail'),
]
