#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <repository_url> <output_file>"
    exit 1
fi

REPO_URL=$1
OUTPUT_FILE=$2

# Create a temporary directory
TEMP_DIR=$(mktemp -d)

# Clone the repository
git clone "$REPO_URL" "$TEMP_DIR"

# Change to the repository directory
cd "$TEMP_DIR"

# Function to check if a file is binary
is_binary() {
    mime=$(file -b --mime-type "$1")
    case "$mime" in
        text/*|application/json|application/xml|application/x-shellscript)
            return 1 ;;
        *)
            return 0 ;;
    esac
}

# Concatenate all text files into a single file
find . -type f -not -path '*/\.*' -print0 | while IFS= read -r -d '' file; do
    if ! is_binary "$file"; then
        echo "Processing: $file"
        echo "--- Start of file: $file ---" >> "$OUTPUT_FILE"
        cat "$file" >> "$OUTPUT_FILE"
        echo -e "\n--- End of file: $file ---\n" >> "$OUTPUT_FILE"
    else
        echo "Skipping binary file: $file"
    fi
done

# Move the output file to the original directory
mv "$OUTPUT_FILE" "$OLDPWD"

# Clean up: remove the temporary directory
cd "$OLDPWD"
rm -rf "$TEMP_DIR"

echo "Repository text contents have been concatenated into $OUTPUT_FILE"
