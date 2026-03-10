from django.shortcuts import render
from django.utils import timezone
from datetime import timedelta

from .models import ABTestRun


AUTO_COMPLETE_AFTER_DAYS = 7


def _refresh_ab_tests():
    changed_fields = []
    now = timezone.now()
    auto_complete_before = now - timedelta(days=AUTO_COMPLETE_AFTER_DAYS)

    for test in ABTestRun.objects.all():
        update_fields = []

        # Keep derived metric synchronized with control vs treatment error values.
        old_improvement = test.improvement_pct
        test.recalculate_improvement()
        if abs(float(old_improvement) - float(test.improvement_pct)) > 1e-9:
            update_fields.append("improvement_pct")

        # Auto-complete old active tests and derive winner from observed MSE.
        if test.status == ABTestRun.STATUS_ACTIVE and test.start_date <= auto_complete_before:
            test.status = ABTestRun.STATUS_COMPLETED
            test.end_date = now
            update_fields.extend(["status", "end_date"])

        if test.status != ABTestRun.STATUS_ACTIVE and not test.end_date:
            test.end_date = now
            update_fields.append("end_date")

        if test.status != ABTestRun.STATUS_ACTIVE and not test.winner_model_version:
            test.winner_model_version = test.resolve_winner()
            update_fields.append("winner_model_version")

        if update_fields:
            test.save(update_fields=sorted(set(update_fields)))
            changed_fields.append((test.id, sorted(set(update_fields))))

    return changed_fields

def ab_testing_index(request):
    """
    Displays current and past A/B tests between model versions.
    """
    _refresh_ab_tests()

    active_tests = ABTestRun.objects.filter(status=ABTestRun.STATUS_ACTIVE).order_by("-start_date")
    past_tests = ABTestRun.objects.exclude(status=ABTestRun.STATUS_ACTIVE).order_by("-end_date", "-start_date")

    context = {
        'active_tests': active_tests,
        'past_tests': past_tests,
    }
    return render(request, 'dashboard/ab_testing_index.html', context)
