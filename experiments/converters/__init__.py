"""Stimulus converters: raw ``data/<study>/`` -> the harness's common schema.

Each module exposes ``convert()``, a generator of :class:`common.StimRow`.  Adding a dataset
means writing one such module and adding it to :data:`CONVERTERS`; ``build_stimuli.py`` needs no
other change.

See ``common.py`` for the read-only / blind-to-human-data / hashed-sources rules every converter
follows, and ``experiments/README.md`` for the schema itself.
"""

from . import chen2023, clark2026, common, gibson2013, huang2024, moses, qian2023, ryskin2021, tabor2004

#: dataset name -> its ``convert()``.  Insertion order is the order ``build_stimuli.py`` builds in.
CONVERTERS = {
    "gibson2013": gibson2013.convert,
    "chen2023": chen2023.convert,
    "ryskin2021": ryskin2021.convert,
    "qian2023": qian2023.convert,
    "huang2024": huang2024.convert,
    "clark2026": clark2026.convert,
    "tabor2004": tabor2004.convert,
    "moses": moses.convert,
}

__all__ = ["CONVERTERS", "common"]
