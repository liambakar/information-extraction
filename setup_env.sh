#!/bin/bash

PROJECT_ROOT="$(pwd)"
echo "project root --------- $PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "PYTHONPATH updated: $PYTHONPATH"