# -*- coding: utf-8 -*-
# Generated by Django 1.9.4 on 2016-03-27 15:34
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stravaChimp', '0004_auto_20160327_1532'),
    ]

    operations = [
        migrations.AlterField(
            model_name='activity',
            name='act_id',
            field=models.IntegerField(default=0, unique=True),
        ),
    ]
