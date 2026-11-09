#!/usr/bin/env bash
# Build the macOS app and install it to /Applications, signed with a stable local identity
# so privacy permissions survive reinstalls. Run ./scripts/create-signing-cert.sh once first.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${ECHO_SIGN_IDENTITY:-Echo Local Signing}"
DEST="${ECHO_INSTALL_DIR:-/Applications}/Echo.app"

if pgrep -f "$DEST/Contents/MacOS/Echo" >/dev/null; then
  echo "Echo is running. Quit it (menu bar → Quit Echo) and run this again." >&2
  exit 1
fi

npm run build
npx electron-builder --config electron-builder.json --mac --dir -c.mac.identity=null
APP="$(ls -d release/mac*/Echo.app | head -1)"

rm -rf "$DEST"
ditto "$APP" "$DEST"
xattr -dr com.apple.quarantine "$DEST" 2>/dev/null || true

hash="$(security find-identity -p codesigning | awk -v n="\"$NAME\"" '$0 ~ n { print $2; exit }')"
if [[ -n "$hash" ]]; then
  codesign --force --deep -s "$hash" "$DEST"
  echo "Installed $DEST (signed with \"$NAME\")."
else
  codesign --force --deep -s - "$DEST"
  echo "Installed $DEST with an ad-hoc signature: macOS will ask for permissions again after every reinstall." >&2
  echo "Run ./scripts/create-signing-cert.sh once to avoid that." >&2
fi
