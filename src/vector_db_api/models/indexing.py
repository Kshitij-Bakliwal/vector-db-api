from typing import Optional
from typing_extensions import Literal

from pydantic import BaseModel, Field

class IndexType(BaseModel):

    type: Literal["flat", "lsh", "ivf"] = "flat"
    # optional per-index params; keep simple for v1
    lsh_num_tables: Optional[int] = Field(8, ge=1, le=64)
    lsh_hyperplanes_per_table: Optional[int] = Field(16, ge=1, le=64)
    ivf_num_centroids: Optional[int] = None
    ivf_nprobe: Optional[int] = None