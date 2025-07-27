#!/bin/bash

# Check if a folder name was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 folder_name"
    exit 1
fi

FOLDER_NAME="$1"

# Create the folder if it doesn't exist
if [ ! -d "$FOLDER_NAME" ]; then
    mkdir -p "$FOLDER_NAME"
fi

# Move Gaussian_*, Record__*, and Xline* files into the folder
mv Gaussian_* "$FOLDER_NAME" 2>/dev/null
mv Record__* "$FOLDER_NAME" 2>/dev/null
mv Xline* "$FOLDER_NAME" 2>/dev/null

# Copy Input__* files into the folder
cp Input__* "$FOLDER_NAME" 2>/dev/null

echo "Files have been processed."

