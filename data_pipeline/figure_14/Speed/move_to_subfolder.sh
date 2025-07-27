#!/bin/bash

# The path to the directory containing the subfolders
parent_dir="./"

# The file you want to copy to all subfolders
file_to_copy="submit_spock.job"

# Find all subdirectories in the parent directory and copy the file into each
find "$parent_dir" -type d -mindepth 1 -exec cp "$file_to_copy" {} \;

