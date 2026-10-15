"""
test_rbac_security.py
DevSecMLOps Security Tests for Student Performance Analytics:
1. Role-Based Access Control (RBAC) permission tests
2. Model cryptographic SHA-256 signature verification
3. Model tampering attack detection
"""

import os
import sys
import tempfile
import pytest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rbac_auth import RBACSecurityManager
from security_scanner import (
    compute_file_sha256,
    sign_model_artifact,
    verify_model_integrity,
    safe_load_model
)
from predict import StudentAnalyticsService


def test_rbac_faculty_allowed():
    assert RBACSecurityManager.is_authorized("faculty", "predict_single") is True
    assert RBACSecurityManager.is_authorized("faculty", "predict_batch") is True


def test_rbac_student_blocked_from_retraining():
    assert RBACSecurityManager.is_authorized("student", "retrain_model") is False
    with pytest.raises(PermissionError):
        RBACSecurityManager.enforce_permission("student", "retrain_model")


def test_rbac_auditor_permissions():
    assert RBACSecurityManager.is_authorized("auditor", "view_audit_logs") is True
    assert RBACSecurityManager.is_authorized("auditor", "predict_single") is False


def test_sha256_model_integrity_valid():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "student_model.joblib")
    checksum_path = model_path + ".sha256"

    assert os.path.exists(model_path)
    assert os.path.exists(checksum_path)
    assert verify_model_integrity(model_path, checksum_path) is True


def test_model_tampering_attack_detection():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"ORIGINAL_STUDENT_MODEL_BINARY_12345")
        temp_model = f.name

    sig_file = temp_model + ".sha256"
    sign_model_artifact(temp_model, sig_file)

    # Attack: tamper with binary
    with open(temp_model, "wb") as f:
        f.write(b"ATTACKER_INJECTED_WEIGHTS_MODIFIED")

    with pytest.raises(PermissionError) as exc_info:
        safe_load_model(temp_model, sig_file)

    assert "Model integrity verification failed" in str(exc_info.value)

    os.remove(temp_model)
    os.remove(sig_file)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
