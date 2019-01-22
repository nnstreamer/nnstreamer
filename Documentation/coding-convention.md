
# Coding Convention

NNStreamer Application follows [The Coding Style of GStreamer](https://gstreamer.freedesktop.org/documentation/frequently-asked-questions/developing.html#what-is-the-coding-style-for-gstreamer-code) for coding convention. 


Basically, the core and almost all plugin modules use K&R with 2-space indenting. Just follow what's already there and you'll be fine. We only require code files to be indented, header may be indented manually for better readability. Please use spaces for indenting, not tabs, even in header files.

When you push your commits, ALWAYS run `gst-indent` to submit a commit with style change.
If there is a change due to code style issues, make two separate commits: (Please do not include other codes' style change in the same commit)
- commit with style change only (i.e., commit gst-indent-formatted original code - not your code change)
- commit with your code change only (i.e., contents only).

```
$ ./tools/development/gst-indent <file-name>
```
