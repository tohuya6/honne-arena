"""Domain errors for the API. main.py maps these to HTTP codes."""


class SubmitError(Exception):
    """Base for user-facing submit failures. `message` is safe to show directly."""

    http_status = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ValidationError(SubmitError):
    http_status = 400


class Unauthorized(SubmitError):
    """Not signed in, or the session expired."""

    http_status = 401


class DuplicateError(SubmitError):
    http_status = 409

    def __init__(self, message: str, existing: dict, rank: int):
        super().__init__(message)
        self.existing = existing
        self.rank = rank


class RateLimited(SubmitError):
    http_status = 429


class QueueFull(SubmitError):
    http_status = 503


class ScoringUnavailable(SubmitError):
    http_status = 503


class ServiceUnavailable(SubmitError):
    """Something other than scoring is down, such as the email relay."""

    http_status = 503


class BackendTimeout(SubmitError):
    """An NDIF call outlived its deadline (e.g. a job stuck in RECEIVED). Its gate slot
    has been released; the abandoned call keeps running in its own thread."""

    http_status = 504


class NotFound(SubmitError):
    http_status = 404


class Forbidden(SubmitError):
    http_status = 403
