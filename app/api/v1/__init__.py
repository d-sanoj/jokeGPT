"""Version 1 of the API: health, corpus, train, generate, experiments."""

from fastapi import APIRouter

from app.api.v1 import corpus, experiments, generate, train

router = APIRouter()
router.include_router(corpus.router)
router.include_router(train.router)
router.include_router(generate.router)
router.include_router(experiments.router)
