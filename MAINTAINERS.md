# Definitions of Roles

## Maintainer

- May override to merge a pull-request or push/revert commits.
- Decides policies and architecture.
- Designate reviewers and their dedicated subdirectories.

## Reviewer

- May reject a pull-request, which prohibits merging.
- May vote for an approval.
- May merge a pull-request if it has enough number of approval-votes from reviewers or maintainers (2 or more).
- The ability (vote/reject/merge) may be limited to a few subdirectories.
- The list of reviewers and their subdirectories is at [.github/CODEOWNERS].

## Committer

- Anyone who has sent a pull-request, which is accepted and merged with the full procedures.

## Note

We allow anyone to sent a pull-request, to provide a code-review (although not being able to vote or reject), or to write an issue as long as they do no harm or break [CODE_OF_CONDUCT.md]
