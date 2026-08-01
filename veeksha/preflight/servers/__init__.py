"""Deterministic mock servers for preflight timing validation.

Each server runs as its own process (see :mod:`veeksha.preflight.spawn`) so its
emit schedule stays punctual and out of the way of the veeksha clients under
test. Servers stamp ground-truth send/receive times and expose them at
``GET /preflight/records``. (Dedicated-core pinning may be added later.)
"""
