#!/bin/bash

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
#echo $CURRENT_DIR
cd $CURRENT_DIR/..
./script/run.sh explorer
