#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
ROOT_DIR=$(git rev-parse --show-toplevel)
TAG=$(git rev-parse --short=6 HEAD)

podman build -t samwelborn/post-process:$TAG -f $SCRIPT_DIR/Containerfile $ROOT_DIR