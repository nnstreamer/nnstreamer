
---
# [Template] PR Description

In general, github system automatically copies your commit message for your convenience.
Please remove unused part of the template after writing your own PR description with this template.
```bash
$ git commit -s filename1 filename2 ... [enter]

Summarize changes in around 50 characters or less

More detailed explanatory text, if necessary. Wrap it to about 72
characters or so. In some contexts, the first line is treated as the
subject of the commit and the rest of the text as the body. The
blank line separating the summary from the body is critical;
various tools like `log`, `shortlog` and `rebase` can get confused 
if you run the two together.

Further paragraphs come after blank lines.

**Changes proposed in this PR:**
- Bullet points are okay, too
- Typically a hyphen or asterisk is used for the bullet, preceded
  by a single space, with blank lines in between, but conventions vary here.

Resolves: #123
See also: #456, #789

**Self evaluation:**
1. Build test: [ ]Passed [ ]Failed [*]Skipped
2. Run test: [ ]Passed [ ]Failed [*]Skipped

**How to evaluate:**
1. Describe how to evaluate in order to be reproduced by reviewer(s).

Add signed-off message automatically by running **$git commit -s ...** command.

$ git push origin <your_branch_name>
```

