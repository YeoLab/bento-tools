Profiling with Scalene

```bash
bento-benchmark → scalene --program-path /Users/ckmah/bento-tools/bento  -m pytest --log-cli-level=DEBUG --log-level=INFO --capture=no tests/_test_measure_points.py::TestMeasurePoints::test_polarity
```

Benchmark notes

Dataset size: 1000 cells x 100 genes x 100 points
- distance: ~1.2 seconds
- polarity: 7 seconds
- moments: 10.5 seconds