from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ab_testing", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="abtestrun",
            name="end_date",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="abtestrun",
            name="winner_model_version",
            field=models.CharField(blank=True, default="", max_length=20),
        ),
        migrations.AlterField(
            model_name="abtestrun",
            name="improvement_pct",
            field=models.FloatField(default=0.0),
        ),
        migrations.AlterField(
            model_name="abtestrun",
            name="status",
            field=models.CharField(
                choices=[
                    ("Active", "Active"),
                    ("Completed", "Completed"),
                    ("Cancelled", "Cancelled"),
                ],
                default="Active",
                max_length=20,
            ),
        ),
    ]
