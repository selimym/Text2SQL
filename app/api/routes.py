from fastapi import APIRouter

router = APIRouter()


def health_check() -> dict[str, str]:
    return {"status": "ok"}


router.add_api_route("/health", health_check, methods=["GET"])
