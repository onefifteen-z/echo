#!/usr/bin/env bash
# Create a self-signed code-signing identity ("Echo Local Signing") in the login keychain.
# Signing local builds with one stable identity keeps macOS privacy grants (Screen Recording,
# Microphone, Speech Recognition) across reinstalls; ad-hoc signatures change on every build.
set -euo pipefail

NAME="${ECHO_SIGN_IDENTITY:-Echo Local Signing}"
KEYCHAIN="$HOME/Library/Keychains/login.keychain-db"

if security find-identity -p codesigning "$KEYCHAIN" | grep -q "\"$NAME\""; then
  echo "Identity \"$NAME\" already exists."
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cat >"$tmp/cert.cnf" <<EOF
[req]
distinguished_name = dn
prompt = no
x509_extensions = ext
[dn]
CN = $NAME
[ext]
basicConstraints = critical,CA:false
keyUsage = critical,digitalSignature
extendedKeyUsage = critical,codeSigning
EOF
openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -config "$tmp/cert.cnf" \
  -keyout "$tmp/key.pem" -out "$tmp/cert.pem" 2>/dev/null
# macOS `security` cannot read OpenSSL 3's default PKCS#12 encryption; use the legacy format.
openssl pkcs12 -export -legacy -name "$NAME" -inkey "$tmp/key.pem" -in "$tmp/cert.pem" \
  -out "$tmp/id.p12" -passout pass:echo
security import "$tmp/id.p12" -k "$KEYCHAIN" -P echo -T /usr/bin/codesign >/dev/null
echo "Created code-signing identity \"$NAME\" (self-signed; \"not trusted\" is expected)."
