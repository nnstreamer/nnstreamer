#!/usr/bin/env bash
echo '#pragma GCC diagnostic push' > $2
echo '#pragma GCC diagnostic ignored "-Wtype-limits"' >> $2
echo '#pragma GCC diagnostic ignored "-Wsign-compare"' >> $2
cat $1 >> $2
echo '#pragma GCC diagnostic pop' >> $2
