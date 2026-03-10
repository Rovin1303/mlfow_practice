from django.db import models


class ABTestRun(models.Model):
    STATUS_ACTIVE = "Active"
    STATUS_COMPLETED = "Completed"
    STATUS_CANCELLED = "Cancelled"
    STATUS_CHOICES = [
        (STATUS_ACTIVE, "Active"),
        (STATUS_COMPLETED, "Completed"),
        (STATUS_CANCELLED, "Cancelled"),
    ]

    test_name = models.CharField(max_length=100)
    control_model_version = models.CharField(max_length=20)
    treatment_model_version = models.CharField(max_length=20)
    control_mse = models.FloatField()
    treatment_mse = models.FloatField()
    improvement_pct = models.FloatField(default=0.0)
    start_date = models.DateTimeField(auto_now_add=True)
    end_date = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_ACTIVE)
    winner_model_version = models.CharField(max_length=20, blank=True, default="")

    def recalculate_improvement(self):
        if self.control_mse and self.control_mse > 0:
            self.improvement_pct = ((self.control_mse - self.treatment_mse) / self.control_mse) * 100.0
        else:
            self.improvement_pct = 0.0

    def resolve_winner(self):
        if self.winner_model_version:
            return self.winner_model_version
        return (
            self.treatment_model_version
            if self.treatment_mse <= self.control_mse
            else self.control_model_version
        )

    def __str__(self):
        return f"{self.test_name} ({self.improvement_pct:.2f}% improvement)"
