window.BENCHMARK_DATA = {
  "lastUpdate": 1620414025358,
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
        "date": 1620414024835,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_main.py::test_main_succeeds",
            "value": 1116.6722174344818,
            "unit": "iter/sec",
            "range": "stddev: 0.00011106180578637621",
            "extra": "mean: 895.5179365861432 usec\nrounds: 615"
          },
          {
            "name": "tests/test_main.py::test_version_option",
            "value": 1507.7012251857468,
            "unit": "iter/sec",
            "range": "stddev: 0.00022873009516188872",
            "extra": "mean: 663.2613831542129 usec\nrounds: 1211"
          },
          {
            "name": "tests/test_main.py::test_version_callback",
            "value": 9440.93025019067,
            "unit": "iter/sec",
            "range": "stddev: 0.00002285185867281921",
            "extra": "mean: 105.92176549337434 usec\nrounds: 2017"
          }
        ]
      }
    ]
  }
}