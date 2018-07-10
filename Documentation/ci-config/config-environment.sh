#!/usr/bin/env bash

##
# @file config-environment.sh
# @brief Environment file to control all scripts commonly for CI bot
#
# This script to maintain consistently all scripts files via this file. 
# In the near future, all configuration variables will be integrated into this file. 
#
# In case that you have to run this CI script at the below environment, Please change
# the contents appropriately.
# a. In case that you want to apply this CI script to another repository
# b. In case that you have to install CI in a new CI server for more high-performance
# c. In case that you need to create new project
#

################# Modify the below statements for your server  #######################


#### Repository setting
# Project name of github.sec.samsung.net
PRJ_REPO_UPSTREAM="nnstreamer"

# CI Server webaddress. Should end with /
CISERVER="http://aaci.mooo.com/"

# Format area (pr-format)
# Add root path of source folders
# Specify a path of source code
# 1) to check prohibited hardcoded paths (e.g., /home/* for now)
# 2) to check code formatting sytele with clang-format
SRC_PATH="./gst/"

# Audit Area (pr-audit)
# Skip build-checker / unit-test checker if all changes are limited to:
# The path starts without / and it denotes the full paths in the git repo. (regex)
SKIP_CI_PATHS="^ci/.*|^Documentation/.*|^\.github/.*|^obsolete/.*|^README\.md|^temporal-bin/.*|^external/.*"

# Define the number of CPUs to build source codes in parallel
# We recommend that you define appropriate # of CPUs that does not result in
# Out-Of-Memory and Too mnay task migration among the CPUs.
CPU_NUM=3

#### Automatic PR commenter: enabling(1), disabling(0)

# inform PR submitter of a rule to pass the CI process
pr_comment_notice=1

# inform all developers of their activity whenever PR submitter resubmit their PR after applying comments of reviews
pr_comment_pr_updated=0

# inform PR submitter that they do not have to merge their own PR directly.
pr_comment_self_merge=0

# infrom PR submitter of how to submit a PR that include lots of commits.
pr_comment_many_commit=0

# inform PR submitter of the webpage address in order that they can monitor the current status of their PR.
pr_comment_pr_monitor=0

#### Build test: Write build type to test ex) "x86_64 armv7l" 
pr_build_arch_type="x86_64 armv7l"

### Check level of doxygen tag:
# Basic = 0 (@file + @brief)
# Advanced = 1 (Basic + "@author, @bug and functions with ctags")
pr_doxygen_check_level=1

#### File size limit
# unit of the file size is MB.
filesize_limit=5

# folder name to exclude the file size check rule. (e.g., <github-repository-name>/temproal-bin/)
filesize_limit_exception_folder="temporal-bin"

################# Do not modify the below statements #################################

# Connecting to a repository using token id instead of git.bot.sec@samsung.com id
# because of two-authentification. Refer to https://github.sec.samsung.net/settings/tokens
TOKEN="01eec554abcaae8755c06c2b06f5d6bb84d4b4a5"

# Email-address
# Note that we have to log-in at least 3 times per a month to avoid deletion of the ID
# according to announcement of "전자녹스포탈" (knoxportal.sec@samsung.com).
EMAIL="git.bot.sec@samsung.com"

# Reference repository to speed up "git clone" command
REFERENCE_REPOSITORY="/var/www/html/$PRJ_REPO_UPSTREAM/"

# RPM repo cache for GBS build
REPOCACHE="/var/www/html/$PRJ_REPO_UPSTREAM/repo_cache/"

# Github repostiroy webaddress
REPOSITORY_WEB="https://github.sec.samsung.net/STAR/$PRJ_REPO_UPSTREAM"
REPOSITORY_GIT="https://github.sec.samsung.net/STAR/$PRJ_REPO_UPSTREAM.git"

# Github webhook API
GITHUB_WEBHOOK_API="https://github.sec.samsung.net/api/v3/repos/STAR/$PRJ_REPO_UPSTREAM"




