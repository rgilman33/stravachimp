# -*- coding: utf-8 -*-
# Generated by Django 1.9.4 on 2016-03-27 03:17
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Activity',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('act_id', models.IntegerField(default=0)),
                ('name', models.CharField(default=b'oosdfa', max_length=50)),
                ('description', models.CharField(default=b'nice run', max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='Athlete',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(default=b'ndsaf', max_length=50)),
                ('athleteId', models.IntegerField(default=0)),
            ],
        ),
    ]
