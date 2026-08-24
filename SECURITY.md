# Security Policy

Please report suspected vulnerabilities privately to **zhouhaoyu@linkerbot.cn**.
Do not open a public GitHub issue for security reports. We aim to acknowledge
reports within 5 business days and to coordinate disclosure thereafter.

## Network listeners

Mirror's optional TCP JSONL and WebSocket listeners are loopback-only and
provide neither authentication nor TLS. Do not expose them to untrusted
networks; tunnel over SSH if remote access is required. Treat any deployment
that binds these listeners to a non-loopback address as a misconfiguration.
