from dataclasses import dataclass
from typing import List

@dataclass
class ActHistogram:
    data: dict
    hook_handles: dict
    accepted_module_name_patterns: List[str]

    def reset(self):
        self.data = {}
        self.hook_handles = {}
