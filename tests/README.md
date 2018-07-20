# How to measure performance of testcases.

This documents is for describing how to measure performance of testcase.

Three features will be supported.

- Tracing(TBD)
- Debugging
- Profiling(TBD)

refer: https://github.sec.samsung.net/STAR/nnstreamer/issues/67

## Tracing

Not implemented yet.

## Debugging

### Pre-requirements

- eog
- gsteamer1.0-tools
- graphviz(dot)

```bash
sudo apt install graphviz
```

- Build with cmake in nnstreamer/build folder.

### How to

```bash
nnstreamer/test$ ./testAll.sh 1
nnstreamer/test$ cd performance/debug/tensor_convertor
$ eog ${number_of_test_case}.png
```

then, you can see elements and caps in pipeline.

## Profiling

Not implemented yet.
