"""
Rate limit for password changes.

WHY MIDDLEWARE AND NOT A VIEW. Password change is allauth's endpoint
(`/api/auth/browser/v1/account/password/change`) and reimplementing it here
to add a limit would mean owning session invalidation, the current-password
check and the reauthentication rules — three things that are easy to get
subtly wrong and that allauth already gets right. So the endpoint stays
theirs and the limit wraps it.

Username and email limits are NOT here: those are our own endpoints in
accounts.py and enforce their limits inline, where the check sits next to the
write it guards.
"""

from django.http import JsonResponse
from django.utils import timezone

PASSWORD_CHANGE_PATH = "/api/auth/browser/v1/account/password/change"


class PasswordChangeRateLimit:
    """Once every `PASSWORD_CHANGE_EVERY`, per account.

    The clock is stamped only on a SUCCESSFUL change (2xx). A rejected
    attempt — wrong current password, too-short new one — must not burn the
    allowance, or one typo locks someone out of their own password for a day.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not self._applies(request):
            return self.get_response(request)

        from .models import Profile

        profile, _ = Profile.objects.get_or_create(user=request.user)
        retry = profile.password_retry_after()
        if retry:
            return JsonResponse(
                {"error": "You can change your password again later.",
                 "retry_after": retry},
                status=429)

        response = self.get_response(request)

        if 200 <= response.status_code < 300:
            profile.password_changed_at = timezone.now()
            profile.save(update_fields=["password_changed_at"])
        return response

    @staticmethod
    def _applies(request) -> bool:
        return (
            request.method == "POST"
            and request.path.rstrip("/") == PASSWORD_CHANGE_PATH
            and getattr(request.user, "is_authenticated", False)
        )
