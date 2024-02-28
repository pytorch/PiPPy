#!/bin/bash

function usage() {
  echo 2>&1 <<EOF
USAGE: ./check [--keep-going] [--pyre]

  --keep-going (default: 1)
  Continue processing even when errors are ecountered.

  --pyre
  Run pyre checks.

  --skip-format
  Don't run format checks.
EOF
}

SKIP_FORMAT=0
SKIP_PYRE=1
KEEP_GOING=1
for x in "$@"; do
  case "$x" in
    '--keep-going')
      KEEP_GOING=1
      ;;

    '--pyre')
      SKIP_PYRE=0
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
  echo; echo "Running format check ..."
  ufmt diff pippy/*.py test/*.py
  (( RETVAL |= $? ))
fi

if (( SKIP_PYRE == 0 )); then
  echo; echo "Running pyre ..."
  pyre check
  (( RETVAL |= $? ))
fi

echo; echo "Running flake8 ..."
flake8 pippy
(( RETVAL |= $? ))

echo; echo "Running mypy ..."
mypy --follow-imports=skip pippy
(( RETVAL |= $? ))

echo; echo "Running pylint ..."
pylint --disable=all --enable=unused-import $(git ls-files '*.py')
(( RETVAL |= $? ))

exit $RETVAL
