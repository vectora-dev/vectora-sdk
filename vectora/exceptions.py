class VectoraError(Exception):
    """Base exception for all SDK errors."""


class VectoraConfigError(VectoraError):
    """Raised when the SDK is configured incorrectly."""


class VectoraAuthError(VectoraError):
    """Raised when the API key is invalid or expired."""


class VectoraConnectionError(VectoraError):
    """Raised when the SDK cannot reach the Vectora API."""


class VectoraRateLimitError(VectoraError):
    """Raised when the Vectora API rate-limits the caller."""


class VectoraNotFoundError(VectoraError):
    """Raised when a requested Vectora resource does not exist."""


class VectoraServerError(VectoraError):
    """Raised when the Vectora API returns an unexpected server error."""


class ComingSoonError(VectoraError):
    """Raised when a not-yet-shipped SDK surface is accessed."""
