#!/usr/bin/env bash
# Install nginx and enable the Ask Me site. Run with sudo.
set -euo pipefail

CONF="$(cd "$(dirname "$0")" && pwd)/askme.nginx.conf"

if ! command -v nginx >/dev/null; then
  echo "==> installing nginx"
  apt-get update -qq
  apt-get install -y -qq nginx
fi

echo "==> linking site config"
ln -sf "$CONF" /etc/nginx/sites-available/askme
ln -sf /etc/nginx/sites-available/askme /etc/nginx/sites-enabled/askme
rm -f /etc/nginx/sites-enabled/default   # frees :80 for us

echo "==> testing config"
nginx -t

echo "==> reloading"
systemctl enable --now nginx
systemctl reload nginx

echo
echo "nginx is up:"
echo "  public  ->  http://localhost/         (admin blocked)"
echo "  admin   ->  http://localhost:8081/admin"
