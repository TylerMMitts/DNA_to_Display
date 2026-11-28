from .analysis import Analysis
from .genetics import Genetics
from .images import Images
from .intake import Intake
from .qc import QC
from .traits import Traits
from .training import Training

REGISTRY = {s.name: s for s in (Intake(), Images(), Traits(), Genetics(), Training(), QC(), Analysis())}
