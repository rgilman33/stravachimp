# -*- coding: utf-8 -*-
# Generated by Django 1.9.4 on 2016-03-27 15:32
from __future__ import unicode_literals

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('stravaChimp', '0003_activity_totaldist'),
    ]

    operations = [
        migrations.AddField(
            model_name='activity',
            name='avgHr',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='activity',
            name='date',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AddField(
            model_name='activity',
            name='easy',
            field=models.FloatField(default=0.0, max_length=50),
        ),
        migrations.AddField(
            model_name='activity',
            name='hrVar',
            field=models.FloatField(default=0.0, max_length=50),
        ),
        migrations.AddField(
            model_name='activity',
            name='impulse',
            field=models.FloatField(default=0.0, max_length=50),
        ),
        migrations.AddField(
            model_name='activity',
            name='realMiles',
            field=models.FloatField(default=0.0, max_length=50),
        ),
        migrations.AddField(
            model_name='activity',
            name='recovery',
            field=models.FloatField(default=0.0, max_length=50),
        ),
        migrations.AddField(
            model_name='activity',
            name='stamina',
            field=models.FloatField(default=0.0, max_length=50),
        ),
        migrations.AddField(
            model_name='activity',
            name='totalTime',
            field=models.FloatField(default=0.0, max_length=50),
        ),
    ]
