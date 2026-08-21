#!/usr/bin/env bash

# set -e

SKIP_PROMPT=false
SEARCH=""
REPLACE=""

# Parse flags and arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -y)
      SKIP_PROMPT=true
      shift
      ;;
    *)
      if [ -z "$SEARCH" ]; then
        SEARCH="$1"
      elif [ -z "$REPLACE" ]; then
        REPLACE="$1"
      else
        echo "Error: Unexpected argument '$1'"
        echo "Usage: $0 [-y] <search_string> <replace_string>"
        exit 1
      fi
      shift
      ;;
  esac
done

# Validate required arguments
if [ -z "$SEARCH" ] || [ -z "$REPLACE" ]; then
  echo "Usage: $0 [-y] <search_string> <replace_string>"
  exit 1
fi

# Find files tracked by Git that contain the search string
FILES=$(git grep -l "$SEARCH" 2>/dev/null || true)

if [ -z "$FILES" ]; then
  echo "Error: No instances of '$SEARCH' found in repository." >&2
  return 1 2>/dev/null || exit 1
fi

# Count affected files
FILE_COUNT=$(echo "$FILES" | wc -l | tr -d ' ')

echo "Found matches for '$SEARCH' in $FILE_COUNT file(s):"
echo "$FILES" | sed 's/^/  - /'
echo ""

# Prompt user unless -y flag was passed
if [ "$SKIP_PROMPT" = false ]; then
  read -p "Replace all instances of '$SEARCH' with '$REPLACE'? [y/N] " -n 1 -r
  echo ""
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation canceled."
    exit 0
  fi
fi

# Execute replacement
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "$FILES" | xargs sed -i '' "s/[[:<:]]${SEARCH}[[:>:]]/${REPLACE}/g"
else
  echo "$FILES" | xargs sed -i "s/\b${SEARCH}\b/${REPLACE}/g"
fi

echo "Successfully replaced '$SEARCH' with '$REPLACE'!"