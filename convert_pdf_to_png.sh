#!/bin/bash

# Path to the figures directory
figures_dir="figures"

# Loop through each PDF file in the figures directory
for pdf_file in "$figures_dir"/*.pdf; do
    # Check if the file is a PDF
    if [ -f "$pdf_file" ] && [ "${pdf_file##*.}" = "pdf" ]; then
        # Get the base name (e.g., figure_1 from figure_1.pdf)
        base_name=$(basename "$pdf_file" .pdf)

        # Convert PDF to PNG using ImageMagick (convert command)
        convert "$pdf_file" "$figures_dir/$base_name.png"

        echo "Converted $pdf_file to $figures_dir/$base_name.png"
    fi
done

echo "All PDFs have been converted."
