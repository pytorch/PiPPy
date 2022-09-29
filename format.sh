#!/bin/bash

DEFAULT_TARGETS=(
  $( git ls-files | \
	  grep '\.py$' | \
	  grep -v '^examples' | \
	  grep -v '^docs' | \
	  grep -v '^pippy' | \
	  grep -v '^test/test_fx' | \
	  grep -v '^test/fx' )
)

function format() {
  local TARGET="$1"

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

