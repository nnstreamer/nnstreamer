#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-2.1-only

set -eu

# Let the retry loop own the first update after adding the PPA.
sudo add-apt-repository -y -n ppa:nnstreamer/ppa
for attempt in 1 2 3; do
  # APT can warn about a failed download and return success by default.
  if sudo apt-get -o APT::Update::Error-Mode=any update; then
    exit 0
  else
    status=$?
  fi

  if [ "$attempt" -eq 3 ]; then
    exit "$status"
  fi

  delay=$((5 * attempt))
  echo "APT update failed (exit $status, attempt $attempt/3); retrying in ${delay}s." >&2
  sleep "$delay"
done
