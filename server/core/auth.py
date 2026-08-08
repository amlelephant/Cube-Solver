"""
Placeholder authentication.

THIS IS NOT LAUNCH AUTH. It exists so the solve endpoints can be built and
tested against a stable notion of "who is this" before allauth lands, and it
is deliberately the least code that provides that.

A device key is a bearer token with no registration, no password and no
recovery. Anyone who sends `Authorization: Device <key>` becomes that user,
so keys are as good as the client's ability to keep them — which for a
browser is "not at all". That is acceptable for a closed build and wholly
unacceptable for a leaderboard, because impersonating a rival costs nothing.

Replace with django-allauth before any solve is worth anything (TODO.md).
"""

from django.contrib.auth import get_user_model
from rest_framework import authentication, exceptions

User = get_user_model()
PREFIX = "Device "
#: Usernames are derived from the key, so bound their length.
MAX_KEY_LEN = 64


class DeviceKeyAuthentication(authentication.BaseAuthentication):
    """`Authorization: Device <key>` -> a get-or-created user.

    Auto-creating on first sight is what makes this usable without a signup
    flow, and is also exactly why it cannot ship: an endpoint that mints a
    user for any string it is shown has no notion of identity at all.
    """

    keyword = "Device"

    def authenticate(self, request):
        header = request.headers.get("Authorization", "")
        if not header.startswith(PREFIX):
            return None
        key = header[len(PREFIX):].strip()
        if not key or len(key) > MAX_KEY_LEN or not key.isalnum():
            raise exceptions.AuthenticationFailed(
                "Device key must be alphanumeric and at most "
                f"{MAX_KEY_LEN} characters.")
        user, _ = User.objects.get_or_create(
            username=f"device_{key}",
            defaults={"is_active": True},
        )
        return (user, key)

    def authenticate_header(self, request):
        return self.keyword
