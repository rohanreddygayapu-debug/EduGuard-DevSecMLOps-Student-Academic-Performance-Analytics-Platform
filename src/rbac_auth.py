"""
rbac_auth.py
Role-Based Access Control (RBAC) Module:
Restricts model operations, student inference, and retraining based on security role.
"""

from typing import Set

ROLE_PERMISSIONS = {
    "admin": {
        "predict_single",
        "predict_batch",
        "retrain_model",
        "view_audit_logs",
        "view_drift_metrics"
    },
    "faculty": {
        "predict_single",
        "predict_batch",
        "view_drift_metrics"
    },
    "student": {
        "predict_single"  # Self assessment only
    },
    "auditor": {
        "view_audit_logs",
        "view_drift_metrics"
    }
}


class RBACSecurityManager:
    @staticmethod
    def is_authorized(role: str, action: str) -> bool:
        normalized_role = str(role).lower().strip()
        permissions = ROLE_PERMISSIONS.get(normalized_role, set())
        return action in permissions

    @staticmethod
    def enforce_permission(role: str, action: str):
        if not RBACSecurityManager.is_authorized(role, action):
            raise PermissionError(
                f"[RBAC ACCESS DENIED] Role '{role}' is not authorized to execute action '{action}'."
            )
        return True
