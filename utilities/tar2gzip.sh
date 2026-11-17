#!/bin/bash

EXTRACT_DIR="extracted_files"
mkdir -p "$EXTRACT_DIR"
shopt -s nullglob

echo "Starting processing..."

for tar_file in *.tar; do
    if [[ -f "$tar_file" ]]; then
        tar_dir="${EXTRACT_DIR}/${tar_file%.tar}"

        # Force extraction if folder doesn't exist OR if tar is newer than the folder
        if [[ ! -d "$tar_dir" ]] || [[ "$tar_file" -nt "$tar_dir" ]]; then
            echo "Extracting $tar_file ..."
            rm -rf "$tar_dir"          # remove old incomplete folder if any
            mkdir -p "$tar_dir"
            
            if tar -xvf "$tar_file" -C "$tar_dir"; then
                echo "✓ Extraction successful: $tar_file"
            else
                echo "✗ Failed to extract $tar_file" >&2
                continue
            fi
        else
            echo "Skipping $tar_file — already extracted and up to date."
            continue
        fi

        # Decompress .csv.gz files
        for sub_dir in "$tar_dir"/*/; do
            if [[ -d "$sub_dir" ]]; then
                echo "  Processing folder: $(basename "$sub_dir")"
                for csv_gz in "$sub_dir"/*.csv.gz; do
                    if [[ -f "$csv_gz" ]]; then
                        csv="${csv_gz%.gz}"
                        if [[ ! -f "$csv" ]]; then
                            echo "    Decompressing $(basename "$csv_gz")..."
                            gzip -d "$csv_gz"
                        else
                            echo "    Already decompressed: $(basename "$csv")"
                        fi
                    fi
                done
            fi
        done
    fi
done

echo "All done."
