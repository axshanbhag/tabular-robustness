# Small-Data Tabular Robustness

Research presented at the Stanford Data-Centric AI Workshop. The study measures the accuracy and runtime costs of adding irrelevant Gaussian features to tabular classification datasets.

## Artifacts

- `code/` contains the archived timing runner, original benchmark and analysis scripts.
- `results/` contains the saved timing replication and earlier baseline and noise predictions.
- `poster/` contains the numerical summaries and individual research figures.

See `results/replication/manifest.json` and `pip_freeze.txt` in the same directory for the recorded run configuration and environment. The runner supports other experiment modes as well. Use the settings in the saved manifest when comparing against the poster. Timing depends on hardware and software versions.

## Analyze the saved timing run

From the repository root, install the research dependencies listed in `code/requirements-colab.txt` in an isolated Python environment, then run:

```sh
python code/analyze.py results/replication --out /tmp/stanford-timing-analysis
```
