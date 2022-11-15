from typing import List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class ImpulseDict:
    """Simple class to record time spans that a range of given impulse channels will be active.
    This class is used both for impulses in the velocity array of the protocerebral bridge and the direct impulses to directional neurons.
    Args:
      N (int) number of channels
      id (dict) the impulse dict.  Each key (channel) has an array of time pairs: ((time0_on, time0_off), (time1_on, time1_off),...)
    """

    # number of channels:
    N: int = 10
    # impulse dict:
    id: Dict[int, Tuple[Tuple[float]]] = field(default_factory=dict)

    def get_impulse(self, t: float) -> int:
        """find impulse pair containing 't' and return its key"""
        for k, v in self.id.items():
            for vpair in v:
                if vpair[0] <= t and t < vpair[1]:
                    return k
