# -*- coding: utf-8 -*-
# Generated by Django 1.9.4 on 2016-03-27 17:21
from __future__ import unicode_literals

import django.contrib.postgres.fields.jsonb
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('stravaChimp', '0009_activity_timezone'),
    ]

    operations = [
        migrations.AddField(
            model_name='athlete',
            name='runsSummary',
            field=django.contrib.postgres.fields.jsonb.JSONField(null=True),
        ),
    ]
