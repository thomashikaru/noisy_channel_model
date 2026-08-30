"""Run the live genjax-port regression suite in one process (loads the LM once).

    NC_LM=EleutherAI/pythia-70m PYTHONPATH=src python -m genjax_port.tests.run

Certifies the path that is actually in production -- the unified pair-HMM RB-SMC filter:
``test_pairhmm_exact`` (exact-enumeration gates + rejuvenation/dedup parity + multi-token),
``test_pythia_word_caprop`` (the Pythia word-caprop smoke), ``test_unigram`` (the
frequency-aware insertion-cost gate), and ``test_morphology`` (the inflectional edit class and
the comma in the indel insertion pool). Each module's ``test_*`` functions are the assertions;
the modules' own ``main()``/``__main__`` blocks are print-only demos and are NOT run here.

Exits non-zero if any assertion fails. See the package docstring for the pytest alternative.
"""

import sys

from genjax_port import lm_penzai as L
from genjax_port.tests import test_pairhmm_exact as t_exact
from genjax_port.tests import test_pythia_word_caprop as t_pythia
from genjax_port.tests import test_morphology as t_morph
from genjax_port.tests import test_unigram as t_unigram


def _tests(module):
    return [(n, getattr(module, n)) for n in sorted(dir(module)) if n.startswith("test_")]


def main():
    L.load_model()
    failures = 0
    for module in (t_exact, t_pythia, t_unigram, t_morph):
        for name, fn in _tests(module):
            try:
                fn()
                print(f"OK    {module.__name__.split('.')[-1]}.{name}", flush=True)
            except Exception as e:  # noqa: BLE001 -- report and continue
                failures += 1
                print(f"FAIL  {module.__name__.split('.')[-1]}.{name}: {e}", flush=True)
    print(f"\n{'all passed' if not failures else str(failures) + ' FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
