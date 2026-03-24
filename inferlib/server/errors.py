from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from inferlib.server.log import logger


@dataclass(slots=True)
class ApiError(Exception):
    status_code: int
    error_type: str
    code: str
    message: str

    def __str__(self) -> str:
        return self.message


def error_payload(*, error_type: str, code: str, message: str) -> dict[str, dict[str, str]]:
    return {
        "error": {
            "type": error_type,
            "code": code,
            "message": message,
        }
    }


def error_response(error: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content=error_payload(
            error_type=error.error_type,
            code=error.code,
            message=error.message,
        ),
    )


def not_found(*, code: str, message: str) -> ApiError:
    return ApiError(
        status_code=404,
        error_type="not_found_error",
        code=code,
        message=message,
    )


def invalid_request(*, code: str, message: str) -> ApiError:
    return ApiError(
        status_code=400,
        error_type="invalid_request_error",
        code=code,
        message=message,
    )


async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return error_response(exc)


async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    first_error = exc.errors()[0] if exc.errors() else {}
    location = ".".join(str(part) for part in first_error.get("loc", []) if part != "body")
    message = first_error.get("msg", "Invalid request")
    if location:
        message = f"{location}: {message}"
    return error_response(
        invalid_request(
            code="validation_error",
            message=message,
        )
    )


async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled server exception")
    return error_response(
        ApiError(
            status_code=500,
            error_type="internal_server_error",
            code="internal_error",
            message="Internal server error",
        )
    )
