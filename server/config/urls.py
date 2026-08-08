"""
URL configuration.

Everything the API serves lives under `/api/`, because that is the prefix
Caddy routes here (see ../Caddyfile). Anything not matching goes to Next.
Adding a top-level route outside `/api/` will 404 in production even though
it works with `runserver` — the proxy never forwards it.
"""

from django.contrib import admin
from django.urls import include, path

from core import accounts, views

urlpatterns = [
    path("api/health/", views.health, name="health"),
    path("api/waitlist/", views.waitlist, name="waitlist"),
    path("api/waitlist/unsubscribe/<str:token>/", views.unsubscribe,
         name="unsubscribe"),
    path("api/scrambles/", views.issue_scramble, name="scrambles"),
    path("api/solves/", views.solves, name="solves"),
    # Before nothing else, but note it must stay a distinct path rather than
    # a query parameter on `solves/`: the two return different shapes and
    # `solves/` is throttled as a submission endpoint.
    path("api/solves/analysis/", views.solve_analysis, name="solve-analysis"),
    path("api/solves/<int:solve_id>/analysis/", views.solve_analysis_detail,
         name="solve-analysis-detail"),

    # Account + profile. `users/<username>/` and `leaderboard/` are public:
    # every profile is viewable by anyone, which is what makes a name on the
    # leaderboard something you can click through to.
    path("api/me/", accounts.me, name="me"),
    path("api/me/username/", accounts.change_username, name="change-username"),
    path("api/me/email/", accounts.change_email, name="change-email"),
    path("api/users/<str:username>/", accounts.public_profile, name="profile"),
    path("api/leaderboard/", accounts.leaderboard, name="leaderboard"),
    # allauth headless: login/signup/verify/reset as JSON under
    # /api/auth/browser/v1/*. "browser" is the session-cookie client type,
    # which is what we want on a single origin — see settings.
    path("api/auth/", include("allauth.headless.urls")),
    # Deliberately under /api/ so the single proxy rule covers it.
    path("api/admin/", admin.site.urls),
]
