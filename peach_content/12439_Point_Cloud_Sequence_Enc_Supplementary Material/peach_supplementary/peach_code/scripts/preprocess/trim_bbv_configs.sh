#!/bin/bash

ROOT="$1"

find "$ROOT" -maxdepth 1 -type d -name "material*" | while read -r dir; do
    config="$dir/config.yaml"
    if [ -f "$config" ]; then
        cp "$config" "$config.bak"
        tail -n +80004 "$config" > "$config.tmp" && mv "$config.tmp" "$config"
        echo "Processed: $config"
    fi
done