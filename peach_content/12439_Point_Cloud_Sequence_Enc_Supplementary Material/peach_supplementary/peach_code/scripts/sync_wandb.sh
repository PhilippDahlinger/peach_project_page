#!/bin/bash

# Recursively find wandb directories and sync them
# Usage: ./sync_wandb.sh [starting_directory]
# If no directory specified, starts from current directory

START_DIR="${1:-.}"

find_and_sync_wandb() {
    local dir="$1"

    # Check if wandb directory exists in current directory
    if [ -d "$dir/wandb" ]; then
        echo "Found wandb directory in: $dir"
        (cd "$dir" && wandb sync --sync-all)
        return 0  # Stop recursion for this branch
    fi

    # Recursively search subdirectories
    for subdir in "$dir"/*; do
        # Check if it's a directory and not a symlink
        if [ -d "$subdir" ] && [ ! -L "$subdir" ]; then
            find_and_sync_wandb "$subdir"
        fi
    done
}

# Start the recursion
find_and_sync_wandb "$START_DIR"