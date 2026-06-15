"""Run all genjax-port regression tests in one process (loads the LM once).

    NC_LM=EleutherAI/pythia-70m PYTHONPATH=. python -m src.genjax_port.tests.run

Exits non-zero on the first failure. See the package docstring for the pytest alternative.
"""

import sys

from src.genjax_port import lm_penzai as L
from src.genjax_port.tests import test_lm_genjax as t1
from src.genjax_port.tests import test_noisy_channel as t2
from src.genjax_port.tests import test_word_model as t3
from src.genjax_port.tests import test_smc_substitution as t4
from src.genjax_port.tests import test_rejuvenation as t5
from src.genjax_port.tests import test_rejuv_bridge as t6
from src.genjax_port.tests import test_rejuv_model as t7


def _tests(module):
    return [(n, getattr(module, n)) for n in sorted(dir(module)) if n.startswith("test_")]


def main():
    L.load_model()
    failures = 0
    for module in (t1, t2, t3, t4, t5, t6, t7):
        for name, fn in _tests(module):
            try:
                fn()
                print(f"OK    {module.__name__.split('.')[-1]}.{name}")
            except Exception as e:  # noqa: BLE001 -- report and continue
                failures += 1
                print(f"FAIL  {module.__name__.split('.')[-1]}.{name}: {e}")
    print(f"\n{'all passed' if not failures else str(failures) + ' FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
