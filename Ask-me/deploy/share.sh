#!/usr/bin/env bash
# Start Ask Me behind nginx and open a public ngrok tunnel.
#
#   ./deploy/share.sh
#
# Stop with Ctrl-C; the tunnel and the app both go down.
set -euo pipefail

cd "$(dirname "$0")/.."
VENV=.venv/bin

if ! command -v nginx >/dev/null; then
  echo "nginx is not installed. Run:  sudo ./deploy/setup-nginx.sh" >&2
  exit 1
fi

if ! pgrep -f "gunicorn -c gunicorn.conf.py" >/dev/null; then
  echo "==> starting gunicorn on 127.0.0.1:8000"
  $VENV/gunicorn -c gunicorn.conf.py wsgi:app &
  GUNICORN_PID=$!
  trap 'kill $GUNICORN_PID 2>/dev/null || true' EXIT
  sleep 4
fi

until curl -sf 127.0.0.1:8000/healthz >/dev/null; do sleep 1; done
echo "==> app healthy: $(curl -s 127.0.0.1:8000/healthz)"

# Confirm the public listener really does hide the panel before we expose it.
code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1/admin)
if [ "$code" != "404" ]; then
  echo "REFUSING TO TUNNEL: /admin returns $code on :80, expected 404." >&2
  echo "Check deploy/askme.nginx.conf is the active site." >&2
  exit 1
fi
echo "==> /admin is blocked on the public listener (404) — safe to tunnel"

echo "==> opening tunnel"
ngrok http 80
