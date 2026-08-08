from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        # Registers the post_save hook that gives every account a Profile.
        # Imported here rather than at module scope because signals touch
        # models, which are not loaded when this module first executes.
        from . import signals  # noqa: F401
