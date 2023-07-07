#!/bin/bash

# USAGE: ./format.sh [--show-targets] [--check] [TARGETS]
# When used with --show-targets, list all default targets and exits.
# When used with --check, reports errors but changes nothing.

DEFAULT_TARGETS=()
for f in $(git ls-files | grep '\.py$'); do
  case "$f" in
    'pippy/fx/'*)
      # ignore
      ;;

    'pippy/'*)
      DEFAULT_TARGETS+=( "$f" )
      ;;

    'examples/'*)
      # ignore
      ;;

    'docs/'*)
      # ignore
      ;;

    'test/'*fx*)
      # ignore
      ;;

    *)
      # include
      DEFAULT_TARGETS+=( "$f" )
      ;;
  esac
done

function format() {
  local TARGET="$1"

  # TODO: enable autoflake and isort.
  # these are not currently enabeled because the existing
  # import structure has magic side-effects that need to
  # be cleaned up so that isort and autoflake don't break them.

  # | autoflake \
  #   --stdin-display-name "$TARGET" \
  #   --remove-all-unused-imports \
  #   - \
  # | isort \
  #   --filename "$TARGET" \
  #   - \

  cat "$TARGET" \
    | black \
      -q \
      --stdin-filename "$TARGET" \
      -

  return ${PIPESTATUS[-1]}
}

function format_check() {
  local TARGET="$1"
  local TFILE=$(mktemp)
  trap "rm $TFILE" EXIT

  format "$TARGET" > "$TFILE"

  diff -u "$TARGET" "$TFILE"

  return $?
}

function reformat_inplace() {
  local TARGET="$1"
  local TFILE=$(mktemp)
  trap "rm $TFILE" EXIT

  format "$TARGET" > "$TFILE"
  if (( $? )); then
    return $?;
  fi

  diff -q "$TARGET" "$TFILE" > /dev/null
  if (( $? )); then
    cat "$TFILE" > "$TARGET";
  fi

  return 0
}


function main() {
  local CHECK
  local TARGETS

  CHECK=0
  TARGETS=()

  for x in "$@"; do
    case "$x" in
      '--show-targets')
  for f in "${DEFAULT_TARGETS[@]}"; do
    echo $f;
  done
  exit 0;
        ;;

      '--check')
        CHECK=1;
        ;;

      *)
        TARGETS+=( "$x" )
        ;;
    esac
  done

  if (( ${#TARGETS[@]} == 0 )); then
    TARGETS=( ${DEFAULT_TARGETS[@]} ) 
  fi

  PY_TARGETS=()
  for x in "${TARGETS[@]}"; do
    if [[ -d "$x" ]]; then
      PY_TARGETS+=( $(find "$x" -name '*.py' -or -name '*.pyi') )

    elif [[ -f "$x" ]]; then
      case "$x" in
        *.py)
          PY_TARGETS+=( "$x" );
    ;;
      esac
    fi
  done

  if (( $CHECK )); then
    local result
    result=0
    for x in "${PY_TARGETS[@]}"; do
      format_check "$x";
      (( result|=$? ))
    done
    exit $result
  else
    local result
    result=0
    for x in "${PY_TARGETS[@]}"; do
      reformat_inplace "$x";
      (( result|=$? ))
    done
    exit $result
  fi
}

main "$@"

