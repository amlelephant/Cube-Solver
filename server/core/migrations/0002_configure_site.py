"""
Point the django.contrib.sites row at the real deployment.

A fresh database ships SITE_ID=1 as "example.com". allauth reads that row for
anything it renders that is not the subject line (which settings pins
explicitly), so leaving it default means links and copy in auth email refer
to example.com. Derived from SITE_URL so it follows the environment rather
than being hardcoded to one domain.

Idempotent, and safe to re-run against an existing row.
"""

from urllib.parse import urlparse

from django.conf import settings
from django.db import migrations


def set_site(apps, schema_editor):
    Site = apps.get_model("sites", "Site")
    host = urlparse(getattr(settings, "SITE_URL", "")).netloc or "localhost:8080"
    Site.objects.update_or_create(
        pk=getattr(settings, "SITE_ID", 1),
        defaults={"domain": host, "name": "CubeArena"},
    )


def unset_site(apps, schema_editor):
    Site = apps.get_model("sites", "Site")
    Site.objects.filter(pk=getattr(settings, "SITE_ID", 1)).update(
        domain="example.com", name="example.com")


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0001_initial"),
        # Must run after the sites table exists AND after its own data
        # migration inserts the default row, or update_or_create races it.
        ("sites", "0002_alter_domain_unique"),
    ]

    operations = [migrations.RunPython(set_site, unset_site)]
