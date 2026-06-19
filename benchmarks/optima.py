"""Known optimal objective values for SteinLib instances.

SteinLib (https://steinlib.zib.de) publishes proven optima for its instances;
we use them as ground truth to validate correctness and measure our gap.  The
classic ``B`` series (sparse random graphs, |V| = 50..100) is included inline as
the default smoke/benchmark set.  Additional optima can be supplied via a CSV
(``instance,optimum`` per line) passed with ``--optima``.
"""

import csv
import os
from typing import Dict, Optional

# SteinLib "B" series proven optima (instance name without extension -> optimum).
B_SERIES: Dict[str, float] = {
    "b01": 82, "b02": 83, "b03": 138, "b04": 59, "b05": 61, "b06": 122,
    "b07": 111, "b08": 104, "b09": 220, "b10": 86, "b11": 88, "b12": 174,
    "b13": 165, "b14": 235, "b15": 318, "b16": 127, "b17": 131, "b18": 218,
}


def load_optima(csv_path: Optional[str] = None) -> Dict[str, float]:
    """Return the known-optima table, optionally extended/overridden by a CSV."""
    optima = dict(B_SERIES)
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            for row in csv.reader(fh):
                if len(row) < 2 or row[0].lower() in ("instance", "name"):
                    continue
                try:
                    optima[row[0].strip()] = float(row[1])
                except ValueError:
                    continue
    return optima


def optimum_for(name: str, optima: Dict[str, float]) -> Optional[float]:
    """Look up the optimum for an instance name (case-insensitive, ext-stripped)."""
    key = os.path.splitext(os.path.basename(name))[0]
    if key in optima:
        return optima[key]
    return optima.get(key.lower())
