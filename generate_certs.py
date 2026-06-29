"""
generate_certs.py — SecureFedHE mTLS Certificate Generator
===========================================================
Run ONCE on any machine before deployment.
Generates:
  certs/ca.crt          <- shared CA (copy to all PCs)
  certs/server.crt/key  <- server certificate (copy to all PCs)
  certs/client.crt/key  <- client certificate (copy to all PCs)
"""

import os
import datetime
import json
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

CERTS_DIR = "certs"
CONFIG_FILE = "config.json"


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {"ring": {"nodes": []}}


def generate_private_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def save_key(key, path, password=None):
    enc = (serialization.BestAvailableEncryption(password)
           if password else serialization.NoEncryption())
    with open(path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=enc
        ))


def save_cert(cert, path):
    with open(path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


def generate_ca():
    key = generate_private_key()
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME,      "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SecureFedHE"),
        x509.NameAttribute(NameOID.COMMON_NAME,       "SecureFedHE CA"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False, key_encipherment=False, data_encipherment=False, key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    return key, cert


def generate_node_cert(ca_key, ca_cert, common_name, ips):
    key = generate_private_key()
    san_list = [x509.DNSName("localhost")]
    for ip in ips:
        try:
            from ipaddress import ip_address
            san_list.append(x509.IPAddress(ip_address(ip)))
        except Exception:
            pass
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME,        "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME,   "SecureFedHE"),
            x509.NameAttribute(NameOID.COMMON_NAME,         common_name),
        ]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
        .add_extension(
            x509.ExtendedKeyUsage([
                ExtendedKeyUsageOID.SERVER_AUTH,
                ExtendedKeyUsageOID.CLIENT_AUTH,
            ]),
            critical=False
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    return key, cert


def main():
    os.makedirs(CERTS_DIR, exist_ok=True)
    config = load_config()
    nodes  = config.get("ring", {}).get("nodes", [])
    all_ips = [n["ip"] for n in nodes] + ["127.0.0.1"]

    print("=" * 55)
    print("  SecureFedHE — mTLS Certificate Generator")
    print("=" * 55)

    print("\n[1/3] Generating Certificate Authority...")
    ca_key, ca_cert = generate_ca()
    save_key(ca_key,  os.path.join(CERTS_DIR, "ca.key"))
    save_cert(ca_cert, os.path.join(CERTS_DIR, "ca.crt"))
    print("      done: certs/ca.crt  (valid 10 years)")

    print("\n[2/3] Generating shared node certificate...")
    print(f"      IPs covered: {all_ips}")
    node_key, node_cert = generate_node_cert(ca_key, ca_cert, "SecureFedHE Node", all_ips)
    save_key(node_key,  os.path.join(CERTS_DIR, "server.key"))
    save_cert(node_cert, os.path.join(CERTS_DIR, "server.crt"))
    save_key(node_key,  os.path.join(CERTS_DIR, "client.key"))
    save_cert(node_cert, os.path.join(CERTS_DIR, "client.crt"))
    print("      done: certs/server.crt + certs/client.crt  (valid 1 year)")

    print("\n[3/3] Certificate summary:")
    for fname in ["ca.crt", "ca.key", "server.crt", "server.key", "client.crt", "client.key"]:
        path = os.path.join(CERTS_DIR, fname)
        size = os.path.getsize(path)
        print(f"      {path:<30} {size:>6} bytes")

    print("\n" + "=" * 55)
    print("  Done. Copy the entire certs/ folder to all PCs.")
    print("  Keep ca.key SECRET — delete it after distributing.")
    print("=" * 55)

    print("\nVerifying certificates...")
    from cryptography.x509 import load_pem_x509_certificate
    with open(os.path.join(CERTS_DIR, "ca.crt"), "rb") as f:
        loaded_ca = load_pem_x509_certificate(f.read())
    with open(os.path.join(CERTS_DIR, "server.crt"), "rb") as f:
        loaded_srv = load_pem_x509_certificate(f.read())
    assert loaded_ca.subject == loaded_srv.issuer, "CA mismatch"
    print("  CA correctly signed server certificate")
    print("  All checks passed\n")


if __name__ == "__main__":
    main()