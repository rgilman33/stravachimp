from django.db import models
from django.contrib.postgres.fields import JSONField
from django.utils import timezone


class Athlete(models.Model):
    name = models.CharField(max_length=50, null=True)
    athleteId = models.IntegerField(null=True)
    profilePic = models.ImageField(null=True, upload_to='profile_pics')
    city = models.CharField(max_length=50, null=True)
    country = models.CharField(max_length=50, null=True)
    sex = models.CharField(max_length=50, null=True)
    premium = models.NullBooleanField(null=True)
    created_at = models.DateTimeField(null=True)
    updated_at = models.DateTimeField(null=True)
    followers = models.IntegerField(null=True)
    friends = models.IntegerField(null=True)
    email = models.EmailField(null=True)
    weight = models.FloatField(max_length=50, null=True)
    meas_pref = models.CharField(max_length=50, null=True)
    
    runsSummary = JSONField(default={})
    fitLines = JSONField(default={})
    masterList = JSONField(default={})
    mafLastFive = JSONField(default={})
    
    def __unicode__(self):
        return self.name
        
class Activity(models.Model):
   
    athlete = models.ForeignKey(Athlete, null=True)
    act_id = models.IntegerField(null=True)
    name = models.CharField(max_length=50, null=True)  
    description = models.CharField(max_length=50, null=True)
    act_type = models.CharField(max_length=50, null=True)  
    timezone = models.CharField(max_length=50, null=True) 
    
    avgHr = models.IntegerField(null=True)
    hrVar = models.FloatField(max_length=50, null=True)
    realMiles = models.FloatField(max_length=50, null=True)
    recovery = models.FloatField(max_length=50, null=True)
    easy = models.FloatField(max_length=50, null=True)
    stamina = models.FloatField(max_length=50, null=True)
    impulse = models.FloatField(max_length=50, null=True)
    date = models.DateTimeField(null=True)
    totalTime = models.FloatField(max_length=50, null=True)
    totalDist = models.FloatField(max_length=50, null=True)
    climb = models.FloatField(max_length=50, null=True)
    mafScore = models.FloatField(max_length=50, null=True)
    
    
    df = JSONField(null=True)
    fitline = JSONField(null=True)
    
    def __unicode__(self):
        return self.name
        


