"""Tests for the genjax-native noisy-channel port.

These are the de-risking spikes (MIGRATION_PLAN.md §2/§4) frozen as version-controlled
regression tests, so the genjax API patterns the port depends on stay green. The repo has no
pytest setup, so run them as scripts (fast with the small LM)::

    NC_LM=EleutherAI/pythia-70m PYTHONPATH=. python -m src.genjax_port.tests.run

or run one file at a time (``python -m src.genjax_port.tests.test_lm_genjax``). The test
functions are plain ``assert``-based, so ``python -m pytest src/genjax_port/tests`` also works
if pytest is installed. The model is loaded once per process; identity assertions (importance
== chain-rule, joint == manual) are LM-independent, so they hold for any NC_LM.
"""
