from django.db import models

class ABTestRun(models.Model):
    test_name = models.CharField(max_length=100)
    control_model_version = models.CharField(max_length=20)
    treatment_model_version = models.CharField(max_length=20)
    control_mse = models.FloatField()
    treatment_mse = models.FloatField()
    improvement_pct = models.FloatField()
    start_date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default='Active')

    def __str__(self):
        return f"{self.test_name} ({self.improvement_pct:.2f}% improvement)"

class ROIMetric(models.Model):
    model_version = models.CharField(max_length=20)
    period = models.CharField(max_length=20) # e.g. "Last 30 Days"
    simulated_profit_usd = models.FloatField()
    risk_reduction_pct = models.FloatField()
    calculated_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"ROI for Version {self.model_version}: ${self.simulated_profit_usd}"
