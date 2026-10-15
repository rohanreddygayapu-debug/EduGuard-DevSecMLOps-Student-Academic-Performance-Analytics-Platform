"""
security_scanner.py
DevSecMLOps Security Utilities for Student Academic Performance Analytics:
- Cryptographic SHA-256 Model Integrity Verification
- Safe Model Loading & Anti-Poisoning Enforcement
"""

import hashlib
import os
import joblib


def compute_file_sha256(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def sign_model_artifact(model_path: str, checksum_path: str = None) -> str:
    if checksum_path is None:
        checksum_path = model_path + ".sha256"

    digest = compute_file_sha256(model_path)
    with open(checksum_path, "w") as f:
        f.write(digest.strip())
    return digest


def verify_model_integrity(model_path: str, checksum_path: str = None) -> bool:
    if checksum_path is None:
        checksum_path = model_path + ".sha256"

    if not os.path.exists(checksum_path):
        raise FileNotFoundError(f"[SECURITY ERROR] Checksum signature missing: {checksum_path}")

    with open(checksum_path, "r") as f:
        expected = f.read().strip()

    actual = compute_file_sha256(model_path)
    if expected != actual:
        raise PermissionError(
            f"[SECURITY CRITICAL] Model integrity verification failed! Expected {expected[:16]}..., got {actual[:16]}..."
        )
    return True


def safe_load_model(model_path: str, checksum_path: str = None):
    verify_model_integrity(model_path, checksum_path)
    return joblib.load(model_path)
