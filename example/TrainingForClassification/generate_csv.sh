#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <base_folder> <output_csv>"
  exit 1
fi

BASE="$1"
CSV="$2"

echo "filename,label" > "$CSV"

for folder in "$BASE"/*/; do
  label=$(basename "$folder")
  for file in "$folder"*.mrc; do
    filename=$(basename "$file" .mrc)
    echo "$filename,$label" >> "$CSV"
  done
done

