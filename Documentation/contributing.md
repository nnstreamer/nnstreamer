---
title: Contributing
...

# How to Contribute

## Coding Convention
Consistent code conventions are important for several reasons:
* *Most importantly:* To make it easy to read and understand the code. Remember: you may be the only one writing the code initially, but many people will need to read, understand, modify, and fix the code over its lifetime. These conventions attempt to make this task much easier. If the code consistently follows these standards, it improves developer understanding across the entire source base.
* To make it easy to debug the code, with both a system call tracer and GNU debuggers. It should be easy to set breakpoints, view locals, and display and view data structures.
* To attempt to improve code quality through consistency, and requiring patterns that are less likely to result in bugs either initially, or after code modification.

For more information, please refer to [coding-convention.md](https://github.com/nnstreamer/nnstreamer/blob/main/Documentation/coding-convention.md).

For C code, you may use [gst-indent](https://github.com/nnstreamer/nnstreamer/blob/main/tools/development/gst-indent).

For C++ code, you may apply clang-format with the given [.clang-format](https://github.com/nnstreamer/nnstreamer/blob/main/.clang-format).
For C/C++ header files, we do not require strict style rules, but it is recommended to apply such rules.

We do not have explicit and strict styling rules for other programming languages, yet.

## Code Reviews and PRs

You are encouraged to review incoming PRs; regardless whether you are a committer, a designated reviewer, or just a passer-by.

If you are a committer or a reviewer of the specific component, you are obligated to review incoming related PRs within some reasonable time frame.
However, even if you are not a reviewer (designated by committer or submitter), as long as you are in this project, you need to give feedback on PRs especially if you are working on similar topics/components.

The submitter has the first responsibility of keeping the created PR clean and neat (rebase whenever there are merge conflicts), following up the feedback, testing when needed.

### Additional requirements for codes
* Each feature should come with a rich set of test cases that can be executed as unit tests during build. If the feature is more invasive or richer, you need more and richer test cases. Refer to other test cases in /tests directory, which use either GTest or SSAT.
* When new test cases are introduced, the number of new negative test cases should be larger than or equal to the number of new positive test cases.
* For C-code, try to stick with C89.
* For C++-code, try to be compatible with C++11. C++ code should be able to be built optionally. In other words, by disabling C++ build option, we should be able to build the whole system without C++ compilers.
* Avoid introducing additional dependencies of libraries. If you are going to use additional libraries, your codes may be located at /ext/* so that they can be "optional" features.
* If your functions or structs/classes are going to be accessed by other modules or NNStreamer users, provide full descriptions of all entries with Doxygen.
* Passing all the tests of TAOS-CI is a necessary condition, but not a satisfying condition.

### Merge Criteria

A PR is required to meet the following criteria.
* It has passed all the tests defined for TAOS-CI.
    - This includes unit tests and integration tests in various platforms and different static analysis tools.
    - Note that one of the tests includes the "Signed-off-by" check, which means that the author has agreed with [Code of Conduct](https://github.com/nnstreamer/nnstreamer/blob/main/CODE_OF_CONDUCT.md). You may need to refer to later section.
* At least TWO committers (reviewers with voting rights, elected by TSC or other committers) have approved the PR.
    - This is a necessary condition, not sufficient.
    - If the PR touches sensitive codes or may affect wide ranges of components, reviewers will wait for other reviewers to back them up.
    - If the PR is messy, you will need to wait indefinitely to get reviews.
        - Apply general rules of git commits and common senses.
        - Do not write a lengthy commit. Apply a single commit per PR if you are new to the community. Have a single topic per commit. Provide enough background information and references. And so on.
* There is no rejections from any official reviewers.
* There is no pending negative feedbacks (unresolved issues) from reviewers.
* A committer with merging privilege will, then, be able to merge the given PR.


## Signing off commits

Each commit is required to be signed-off by the corresponding author.
With properly configured development environment, you can add sign-off for a commit with ```-s``` option: e.g., ```git commit -s```.
[Here is some stories on sign-off](https://stackoverflow.com/questions/1962094/what-is-the-sign-off-feature-in-git-for)

- How to give the developers zero cost:
```bash
$ vi ~/.gitconfig
  [user]
          name = Gildong Hong
          email = gildong.hong@samsung.com
$ git commit -s <file-name>
// -s (--signoff) means automated signed-off-by statement
```

### What does it mean to sign off commits for authors?

Including a "Signed-off-by:" tag in your commit means that you are making the Developer Certificate of Origin (DCO) certification for that commit. A copy of the DCO text can be found at https://developercertificate.org/

## How to contribute to GStreamer community
* https://gstreamer.freedesktop.org/documentation/contribute/index.html
