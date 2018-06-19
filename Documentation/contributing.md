# How to Contribute

## Coding Convention
Consistent code conventions are important for several reasons:
* *Most importantly:* To make it easy to read and understand the code. Remember: you may be the only one writing the code initially, but many people will need to read, understand, modify, and fix the code over its lifetime. These conventions attempt to make this task much easier. If the code consistently follows these standards, it improves developer understanding across the entire source base.
* To make it easy to debug the code, with both a system call tracer and GNU debuggers. It should be easy to set breakpoints, view locals, and display and view data structures.
* To attempt to improve code quality through consistency, and requiring patterns that are less likely to result in bugs either initially, or after code modification.

## Code Reviews and PRs

Please review incoming PRs; regardless whether you are a maintainer, a designated reviewer, or just a passer-by.

If you are a maintainer or reviewer of the specific component, you are obligated to review incoming related PRs within some reasonable time frame.
However, even if you are not a reviewer (designated by committer or maintainers) or maintainer, as long as you are in this project, you need to give feedback on PRs especially if you are working on similar topics/components.

The submitter has the first responsibility of keeping the created PR clean and neat (rebase whenever there are merge conflicts), following up the feedback, testing when needed.

### Any commits are required to be reviewed and approved before merged.

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

It means that you are legally responsible for the given commit that, according to your best knowledge,
- It is the original work of the author (no plagiarism or copy-pasting from external work)
- It does not have any patents, license, or copyright issues.
- You are granting all related rights to the community or project (depending on the licenses)

Usually, you are not allowed to push commits of other's work; however, you may do so if you
- Maintain the original authorship in the git commit (you can edit authors of a given commit: ```$ man git commit```)
- You can sure there is no patents, licenses, or copyright issues.
- You can grant all related rights to the community or project (depending on the licenses)

From Torvalds' (original author of Linux and git) git repo, (documentation about git commit message tags)[https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/Documentation/process/5.Posting.rst]
> Signed-off-by: this is a developer's certification that he or she has
   the right to submit the patch for inclusion into the kernel.  It is an
   agreement to the Developer's Certificate of Origin, the full text of
   which can be found in Documentation/process/submitting-patches.rst.  Code without a
   proper signoff cannot be merged into the mainline.

## How to contribute to GStreamer community
* https://gstreamer.freedesktop.org/documentation/contribute/index.html
