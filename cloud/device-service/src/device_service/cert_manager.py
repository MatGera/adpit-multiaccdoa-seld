"""Certificate manager for mTLS device provisioning."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

import structlog

if TYPE_CHECKING:
    from device_service.config import Settings

logger = structlog.get_logger(__name__)


class CertificateManager:
    """Manages device certificate generation and signing for mTLS."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ca_cert: x509.Certificate | None = None
        self._ca_key: ec.EllipticCurvePrivateKey | None = None

    def _load_ca(self) -> None:
        """Load the CA certificate and private key."""
        ca_cert_path = Path(self._settings.ca_cert_path)
        ca_key_path = Path(self._settings.ca_key_path)

        if not ca_cert_path.exists() or not ca_key_path.exists():
            logger.warning("ca_certs_not_found, generating self-signed CA")
            self._generate_self_signed_ca()
            return

        with open(ca_cert_path, "rb") as f:
            self._ca_cert = x509.load_pem_x509_certificate(f.read())
        with open(ca_key_path, "rb") as f:
            self._ca_key = serialization.load_pem_private_key(f.read(), password=None)  # type: ignore[assignment]

        logger.info("ca_loaded", subject=str(self._ca_cert.subject))

    def _generate_self_signed_ca(self) -> None:
        """Generate a self-signed CA for development."""
        key = ec.generate_private_key(ec.SECP256R1())

        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "IT"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SELD Digital Twin"),
            x509.NameAttribute(NameOID.COMMON_NAME, "SELD Dev CA"),
        ])

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.UTC))
            .not_valid_after(
                datetime.datetime.now(datetime.UTC)
                + datetime.timedelta(days=3650)
            )
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=0),
                critical=True,
            )
            .sign(key, hashes.SHA256())
        )

        self._ca_cert = cert
        self._ca_key = key

        # Save to disk
        ca_dir = Path(self._settings.ca_cert_path).parent
        ca_dir.mkdir(parents=True, exist_ok=True)

        with open(self._settings.ca_cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        with open(self._settings.ca_key_path, "wb") as f:
            f.write(key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ))

        logger.info("self_signed_ca_generated")

    def issue_device_certificate(self, device_id: str) -> dict:
        """Issue a new client certificate for a device.

        Args:
            device_id: Unique device identifier.

        Returns:
            Dict with PEM-encoded certificate and private key.
        """
        if self._ca_cert is None or self._ca_key is None:
            self._load_ca()

        assert self._ca_cert is not None
        assert self._ca_key is not None

        # Generate device key pair
        device_key = ec.generate_private_key(ec.SECP256R1())

        subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "IT"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SELD Digital Twin"),
            x509.NameAttribute(NameOID.COMMON_NAME, device_id),
        ])

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(self._ca_cert.subject)
            .public_key(device_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.UTC))
            .not_valid_after(
                datetime.datetime.now(datetime.UTC)
                + datetime.timedelta(days=self._settings.cert_validity_days)
            )
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName(device_id),
                ]),
                critical=False,
            )
            .sign(self._ca_key, hashes.SHA256())
        )

        cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode()
        key_pem = device_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode()
        ca_pem = self._ca_cert.public_bytes(serialization.Encoding.PEM).decode()

        logger.info("device_cert_issued", device_id=device_id)

        return {
            "device_id": device_id,
            "certificate": cert_pem,
            "private_key": key_pem,
            "ca_certificate": ca_pem,
        }
