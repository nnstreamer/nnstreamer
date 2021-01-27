---
title: Rank counting
...

### Rank counting with other/tensor type
1. We do NOT declare RANK values in stream capabilities (GST-CAP).
2. Users may omit later parts of dimension expressions if the later parts of dimensions are 1; however, this does NOT imply that the later parts are not counted for RANK values.
    - "10:20:1:1" == "10:20:1" == "10:20"
    - Rank value of the above dimensions may be 2, 3, or 4.
3. If rank values are required by a tensor-filter or tensor-decoder, users may explicitly declare the corresponding rank values with properties (This function is WIP as of 2020-10-06)
    - If not declared explicitly, but the corresponding sub-plugin requires rank value, an internal rank-counting function will count the rank assuming that:
        - rank("10:10:1:1") == rank("10:10") == rank("10:10:1") == 2
        - rank("1:1:1:1") == rank("1") == 1
        - rank("10:10:10:10") == 4

As of 2020-10-07, there appears to be inconsistency in rank-counting, we are working on resolving them.
