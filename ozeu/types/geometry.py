from typing import List
import numpy as np
import dataclasses

@dataclasses.dataclass
class RectROI:
    min_x: int
    min_y: int
    max_x: int
    max_y: int
    contours: List[List[float]]
    
    def get_numpy_slice(self):
        return np.s_[self.min_y:self.max_y, self.min_x:self.max_x]