---
title: How to use Test Cases
...

## How to run Test Cases

- Use the built library as a general gstreamer plugin.

- Unit Test

For gtest based test cases (common library and nnstreamer plugins)
```
$ cd build
$ ninja test
```

For all gst-launch-based test cases ([SSAT](https://github.com/myungjoo/SSAT), mostly golden testing)
```
$ cd tests
$ ssat
```

## How to write Test Cases
* [How to write Test Cases](how-to-write-testcase.md)
