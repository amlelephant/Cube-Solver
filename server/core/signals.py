"""
Every account gets a Profile, always.

A signal rather than "create it in the signup view", because accounts are
created from at least four places that do not share a code path: allauth
headless signup, `manage.py createsuperuser`, the Django admin, and
`manage.py seed_demo`. Anything that misses one produces a user whose
`user.profile` raises `RelatedObjectDoesNotExist` on first page load — and
that failure surfaces far from its cause.

`get_or_create` rather than `create` so this is idempotent: a fixture load or
a re-save of an existing user must not collide.
"""

from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import Profile


@receiver(post_save, sender=settings.AUTH_USER_MODEL,
          dispatch_uid="core.ensure_profile")
def ensure_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.get_or_create(user=instance)
