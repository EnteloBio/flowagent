#!/bin/bash
# Data cleanup utility for genomics data files

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --min_quality)
            MIN_QUALITY="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Validate inputs
if [[ -z "$INPUT_FILE" ]]; then
    echo "Error: --input_file is required" >&2
    exit 1
fi

if [[ -z "$MIN_QUALITY" ]]; then
    MIN_QUALITY=20  # Default minimum quality score
fi

# Create output directory
OUTPUT_DIR="cleaned_data"
mkdir -p "$OUTPUT_DIR"

# Clean data
CLEANED_FILE="$OUTPUT_DIR/cleaned.txt"
STATS_FILE="$OUTPUT_DIR/cleanup_stats.json"

# Process file and gather statistics
TOTAL_LINES=0
KEPT_LINES=0
FILTERED_LINES=0

while IFS= read -r line; do
    ((TOTAL_LINES++))
    
    # Skip comment lines
    if [[ $line =~ ^# ]]; then
        continue
    fi
    
    # Skip empty lines
    if [[ -z "${line// }" ]]; then
        continue
    fi
    
    # Extract quality score (assuming tab-separated format with quality in field 5)
    QUALITY=$(echo "$line" | cut -f5)
    
    # Keep lines meeting quality threshold
    if (( $(echo "$QUALITY >= $MIN_QUALITY" | bc -l) )); then
        echo "$line" >> "$CLEANED_FILE"
        ((KEPT_LINES++))
    else
        ((FILTERED_LINES++))
    fi
done < "$INPUT_FILE"

# Generate statistics
cat > "$STATS_FILE" << EOF
{
    "total_lines": $TOTAL_LINES,
    "kept_lines": $KEPT_LINES,
    "filtered_lines": $FILTERED_LINES,
    "retention_rate": $(echo "scale=4; $KEPT_LINES / $TOTAL_LINES" | bc)
}
EOF

# Sort and deduplicate the cleaned file
sort -u "$CLEANED_FILE" -o "$CLEANED_FILE"

# Output results as JSON
echo "{
    \"cleaned_file\": \"$CLEANED_FILE\",
    \"stats_file\": \"$STATS_FILE\"
}"
