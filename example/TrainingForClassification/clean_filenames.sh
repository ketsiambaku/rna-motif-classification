#!/bin/bash

# Remove "1x1_" from anywhere in .pdb filenames
for file in *.pdb; do
    if [[ "$file" == *"1x1_"* ]]; then
        newname="${file//1x1_/}"
        mv "$file" "$newname"
        echo "Renamed $file -> $newname"
    fi
done

# Remove "EMD-<number>_" from anywhere in .mrc filenames
for file in *.mrc; do
    if [[ "$file" =~ EMD-[0-9]+_ ]]; then
        newname="$(echo "$file" | sed -E 's/EMD-[0-9]+_//')"
        mv "$file" "$newname"
        echo "Renamed $file -> $newname"
    fi
done
