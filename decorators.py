"""
Server-side role enforcement. This is the ONLY place access control is
decided - never trust the frontend (hidden buttons, JS checks, etc.) to
keep users out of pages they shouldn't see. Every dashboard/admin/doctor
route in app.py uses this decorator.
"""

from functools import wraps
from flask import abort
from flask_login import current_user


def role_required(*allowed_roles):
    """
    Usage: @role_required("admin")  or  @role_required("doctor", "admin")

    Blocks the request with 401 (not logged in) or 403 (logged in, but
    wrong role) before the view function ever runs - so there is no way
    to reach a role-restricted page's logic by guessing its URL.
    """
    def decorator(view_func):
        @wraps(view_func)
        def wrapped_view(*args, **kwargs):
            if not current_user.is_authenticated:
                abort(401)
            if current_user.role not in allowed_roles:
                abort(403)
            return view_func(*args, **kwargs)
        return wrapped_view
    return decorator
