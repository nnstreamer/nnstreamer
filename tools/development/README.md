---
title: Development tools
...

## Development 

### getTestModels.sh
Get network model for evaluation

### gst-indent
Check for existence of indent, and error out if not present

### pre-commit
Verify what is about to be committed

### reversion.sh
Update version info for packaging, build, ...

#### Usage

```bash
$ ./reversion.sh <old-major> <old-mid> <old-minor> <new-version(full)> <"Name <Email>">
```

### count_test_cases.py
Aggregate unit test result

#### Usage

```bash
$ ./count_test_cases.py <gtest xml path> <ssat summary path>
```

### nnstreamerCodeGenCustomFilter.py
Generate code for nnstreamer custom filters

