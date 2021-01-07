---
title: How to archive large files
...

# How to archive large files with git-lfs
In order to archive a big data such as map, training, network model, and image data,
We recommend that you use `git-lfs`command as a official process.
Note that github does not support uploading big files more than +100MB by default. 
You do not need to operate your own development server to archive big files thanks to `git-lfs` facility.
However, as github.com charges for lfs, we cannot use it in nnsuite/testcases.
Please wait for AWS-nnsuite account activation for large files.

Git Large File Storage (LFS) replaces large files such as audio samples, videos, datasets, and graphics
with text pointers inside Git, while storing the file contents on a remote server like GitHub.com 
or GitHub Enterprise. Note that Git LFS requires Git v1.8.5 or higher.

# Install git-lfs package
Packagecloud (https://packagecloud.io) hosts `git-lfs` packages with apt/deb for popular Linux distribution such as Ubuntu.
```bash
$ curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
$ sudo apt install git-lfs
```
If you are using another Linux distribution such as Fedora, Please refer to the below webpage.
* Installing git-lfs: https://github.com/git-lfs/git-lfs/blob/v2.4.0/INSTALLING.md#installing-on-linux-using-packagecloud

# Case study: TF-Vision_EvaluationSet Repository 
Note that default branch must be `tizen` instead of `master` in case of TAOS. We assume that you have to upload LibreOffice Installer
(251MB) in a github repository.

Clone the github repository with upstream and origin.
```bash
$ mkdir TF-Vision_EvaluationSet
$ cd    TF-Vision_EvaluationSet
$ git remote add upstream GITURL
$ git remote add origin   GITURL
$ git fetch upstream
$ git pull upstream tizen
```

You need to setup the global Git hooks for Git LFS.
```bash
$ git lfs install
$ wget http://mirrors.adams.edu/LibreOffice/win/LibreOffice_6.0.4_Win_x86.msi
```

Add all *.msi files through Git LFS.
```bash
$ git lfs track "*.msi"
```

Now you are ready to push some commits
```bash
$ git add LibreOffice_6.0.4_Win_x86.msi
$ git commit -m "add msi file"
```

You can confirm that Git LFS is managing your *.msi file.
```bash
$ git lfs ls-files
# Push your files to the Git remote 
$ git push origin <your-branch-name>
```

# How to remove large file from commit history

* Method 1:
```bash
$ firefox https://rtyley.github.io/bfg-repo-cleaner/
$ java -jar bfg-x.x.x.jar --strip-blobs-bigger-than 100M
$ git repack && git gc
```

* Method 2:
```bash
$ git filter-branch --index-filter 'git rm --cached --ignore-unmatch *.data' -- --all
$ rm -Rf .git/refs/original
$ rm -Rf .git/logs/
$ git gc --aggressive --prune=now
$ git count-objects -v
```

# Reference
* Getting Started: https://github.com/git-lfs/git-lfs/tree/v2.4.0#getting-started
* Git extension for versioning large files: https://git-lfs.github.com
