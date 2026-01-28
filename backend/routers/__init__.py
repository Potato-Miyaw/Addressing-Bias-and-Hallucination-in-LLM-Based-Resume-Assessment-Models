"""
Routers package initialization
"""

from . import jd_router
from . import resume_router
from . import verification_router
from . import matching_router
from . import ranking_router
from . import pipeline_router
from . import bias_router  # Multi-Model Bias Detection

__all__ = [
    "jd_router",
    "resume_router",
    "verification_router",
    "matching_router",
    "ranking_router",
    "pipeline_router",
    "bias_router"  # Multi-Model Bias Detection
]