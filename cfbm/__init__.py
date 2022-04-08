from .base import *
from .utils import *
from .formed import (
    FFTFormedSincNSBeamModel,
    FFTFormedActualBeamModel,
)
from .composite import (
    CompositeBeamModel,
)

from .config import current_config, current_model_name

current_model_class = globals()[current_model_name]
