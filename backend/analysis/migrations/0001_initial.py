# Generated by Django 5.1 on 2024-08-31 16:01

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Analysis',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.TextField()),
                ('sentiment', models.CharField(max_length=10)),
                ('score', models.FloatField()),
            ],
        ),
    ]
