# Security

The Flask endpoint is a local research service. It binds to loopback, disables debug mode, and limits upload size. Public deployment still needs authentication, rate limiting, isolated jobs, and resource controls.

Load only trusted checkpoints. The maintained loader uses PyTorch's `weights_only=True` mode. Input validation restricts the demo to its documented GeoTIFF contract, but does not make an internet-facing upload service production-ready.

Do not include secrets, private imagery, or access tokens in issues or notebook outputs. To report a sensitive problem, contact the maintainer through the contact method listed on [their GitHub profile](https://github.com/prithvi-kaizen); avoid publishing exploit details before a fix is available.
