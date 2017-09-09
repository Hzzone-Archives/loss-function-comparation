#!/usr/bin/env sh
set -e

TOOLS=/Users/HZzone/caffe/build/tools

$TOOLS/caffe train --solver=./mnist_siamese_solver.prototxt $@
