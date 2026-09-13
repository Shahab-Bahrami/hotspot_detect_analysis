# Regional customer hotspot filter investigation

Open these notebooks from the repository root or `notebooks/`:

1. [01_customer_satellite_matching.ipynb](../01_customer_satellite_matching.ipynb) — source inventory, region map, timezone inference, matching tolerances, provider overlap, and likely satellite-label errors.
2. [02_regional_hotspot_filter_tests.ipynb](../02_regional_hotspot_filter_tests.ipynb) — quoted rule, alternative interpretations, geographic sensitivity, confidence/FRP tests, counterexamples, and exploratory feature reconstruction.

Both contain executed tables and plots, a findings table at the beginning, explanations next to the tests, and conclusions. Each can run independently. Use **Kernel → Restart Kernel and Run All Cells**; running dependent cells out of order can produce `NameError`.

Install dependencies in your preferred Python environment with:

```sh
python -m pip install -r notebooks/hotspot_filter_analysis/requirements.txt
```

The computational functions and synthetic tests are in [analysis.py](analysis.py). All analysis is local. The source workbook and satellite CSVs are unchanged. Caches rebuild when input checksums or helper code change; set `RECOMPUTE = True` to force rebuilding. Notebook narratives describe the supplied September 2026 snapshot and should be reviewed when inputs change.

The study area is the bounding rectangle of the customer locations in Jambi / South Sumatra, plus a halo for neighbor searches. This is an approximation to their monitoring area, not a supplied concession boundary. Proximity-to-customer-location masks are sensitivity analyses and are not independent geographic validation.

Results are in [outputs/hotspot_filter_analysis](../outputs/hotspot_filter_analysis/). Particularly useful files:

- `customer_match_audit.csv`: every customer row with its chosen archive match or missing match.
- `strong_satellite_label_disagreements.csv`: conservative alternative-source matches, including 753 likely Aqua/Terra labeling errors in 2023.
- `customer_rule_audit.csv`: matched customer rows and cluster-rule outcomes.
- `high_confidence_rule_counterexamples.csv`: unambiguous, close matches isolated even under the 2 km interpretation.
- `regional_detection_audit.csv.gz`: regional archive rows, features, observed-list membership, and rule outcome.
- `quoted_rule_comparison.csv`, `candidate_rule_sweep.csv`, `exploratory_tree_metrics.csv`: aggregate tests.
- `label_correction_rule_sensitivity.csv`, `label_correction_tree_sensitivity.csv`: original labels versus likely corrections.
- `input_inventory.csv`, `analysis_provenance.json`, `tested_environment.json`: checksums, settings, and tested package versions.

The quote alone does not reconstruct the list. Geography and provider/product handling are better-supported additional factors; the exact original algorithm is not identifiable without monitoring boundaries and row-level alert flags. List-membership metrics must not be read as fire-detection precision or recall.
