# research

Archived research and one-off modules moved out of the `worldcup` package. Not part of the pipeline and not installed as entry points.

Run from the repo root with the research folder on the path, for example:

    PYTHONPATH=research uv run python research/sensitivity.py
    PYTHONPATH=research uv run python research/scripts/sweep_host_bonus.py

Modules here may depend on data files that the slimmed pipeline no longer rebuilds (rosters, trophy scores, club badges). FBref stats are not included: to run `fbref_loader.py` or `squad_strength.py`, save your own copy to `data/raw/fbref/` and check FBref's terms first.
