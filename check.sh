#!/bin/bash

function usage() {
  echo 2>&1 <<EOF
USAGE: ./check [--keep-going] [--skip-pyre]

  --keep-going (default: 0)
  Continue processing even when errors are ecountered.

  --skip-pyre
  Don't run pyre checks.

  --skip-format
  Don't run format checks.
EOF
}

SKIP_FORMAT=0
SKIP_PYRE=0
KEEP_GOING=0
for x in "$@"; do
  case "$x" in
    '--keep-going')
      KEEP_GOING=1
      ;;

    '--skip-pyre')
      SKIP_PYRE=1
      ;;

    '--skip-format')
      SKIP_FORMAT=1
      ;;

    *)
      echo "Unknown option: $x" 2>&1
      usage
      exit 1;
      ;;
  esac
done

if (( KEEP_GOING == 0 )); then
  set -e
fi


RETVAL=0

if (( SKIP_FORMAT == 0 )); then
  ./format.sh --check
  (( RETVAL |= $? ))
fi

if (( SKIP_PYRE == 0 )); then
  pyre check
  (( RETVAL |= $? ))
fi

flake8 pippy spmd test/spmd
(( RETVAL |= $? ))

# mypy spmd test/spmd
mypy $(git ls-files '*.py' | grep -v pippy/fx | grep -v test/.*fx | grep -v examples/hf)
(( RETVAL |= $? ))

pylint --disable=all --enable=unused-import $(git ls-files '*.py')
(( RETVAL |= $? ))

exit $RETVAL

