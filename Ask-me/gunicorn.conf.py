"""gunicorn settings for Ask Me."""

import multiprocessing
import os

bind = os.getenv("GUNICORN_BIND", "127.0.0.1:8000")

# PostgreSQL is the shared source of truth, so multiple workers can safely
# serve requests and observe synchronization changes.
workers = int(os.getenv("GUNICORN_WORKERS", "2"))
threads = int(os.getenv("GUNICORN_THREADS", str(min(8, multiprocessing.cpu_count() * 2))))
worker_class = "gthread"

# A Groq call plus embedding can take a while on a cold model load.
timeout = 120
graceful_timeout = 30
keepalive = 5

# Trust the reverse proxy in front of us for X-Forwarded-* headers.
forwarded_allow_ips = os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1")

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")
