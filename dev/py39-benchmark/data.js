window.BENCHMARK_DATA = {
  "lastUpdate": 1620414090748,
  "repoUrl": "https://github.com/TeoZosa/deep-learning-v2-pytorch",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "teofilo@sonosim.com",
            "name": "Teo Zosa",
            "username": "TeoZosa"
          },
          "committer": {
            "email": "teofilo@sonosim.com",
            "name": "Teo Zosa",
            "username": "TeoZosa"
          },
          "distinct": true,
          "id": "76517732bc5d3a6268b2f09f0d2dd57846c3836c",
          "message": ":sparkles: Add template boilerplate",
          "timestamp": "2021-05-07T10:51:39-07:00",
          "tree_id": "db896406a17034d8359d4e03103a358aaf54311c",
          "url": "https://github.com/TeoZosa/deep-learning-v2-pytorch/commit/76517732bc5d3a6268b2f09f0d2dd57846c3836c"
        },
        "date": 1620414089676,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_main.py::test_main_succeeds",
            "value": 883.1637563147507,
            "unit": "iter/sec",
            "range": "stddev: 0.00014573692675606307",
            "extra": "mean: 1.1322928424653445 msec\nrounds: 292"
          },
          {
            "name": "tests/test_main.py::test_version_option",
            "value": 1206.1582279951078,
            "unit": "iter/sec",
            "range": "stddev: 0.00033591912169765863",
            "extra": "mean: 829.078620689935 usec\nrounds: 899"
          },
          {
            "name": "tests/test_main.py::test_version_callback",
            "value": 8452.084277420663,
            "unit": "iter/sec",
            "range": "stddev: 0.00006596709276970044",
            "extra": "mean: 118.3140119262005 usec\nrounds: 1090"
          }
        ]
      }
    ]
  }
}