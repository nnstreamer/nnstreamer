# NNStreamer APIs

## Files
- nnstreamer-capi-pipeline.c - API to make pipeline with NNStreamer
- nnstreamer-capi-single.c - API to run a single model with NNStreamer, independent of GStreamer
- nnstreamer-capi-util.c - Utility functions for capi
- tensor\_filter\_single.c - Tensor\_filter independent of GStreamer

## Comparison of Single API

### Latency & Running-time
Below shows the comparison of old pipeline-based vs Gst-less Single API - showing reduction in latency and running-time for the API (tested with tensorflow-lite).

These values averaged over 10 continuous runs.

|  | Open (us)      | Invoke (ms)           | Close (us)  |
| --- |:-------------:|:-------------:|:-----:|
| New (cache warmup) |  658   | 5195 | 206 |
| Old (cache warmup) |  1228 | 5199   | 5245 |
| New (no warmup) | 1653 | 5205  | 201 |
| Old (no warmup) | 7201 | 5225  | 5299  |


These values are just for the first run.

| | Open (us)      | Invoke (ms)           | Close (us)  |
| --- |:-------------:|:-------------:|:-----:|
| New  |  12326   | 5231 | 2347 |
| Old |  58772 | 5250   | 52611 |

### Memory consumption

Comparison of the maximum memory consumption between the two API implementations. Both the examples below run the same test case. The difference is that the test case executable has been linked with different API shared library. [mprof](https://github.com/pythonprofilers/memory_profiler) has been used to measure the memory consumption with `--interval 0.01 --include-children` as configuration parameters. The unit test `nnstreamer_capi_singleshot.benchmark_time` was used to compare the memory consumption.

- Old pipeline-based implementation
![unittest_tizen_capi_single_old](https://user-images.githubusercontent.com/6565775/64155170-3e179200-ce6d-11e9-94d5-b3a4138533c7.png)

- Gst-less implementation
![unittest_tizen_capi_single_new](https://user-images.githubusercontent.com/6565775/64155182-4374dc80-ce6d-11e9-9d74-660d6261ceb6.png)
