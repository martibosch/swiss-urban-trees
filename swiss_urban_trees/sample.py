"""Utility functions for select training tiles."""

import numpy as np


def select_balanced_tiles(
    tile_data: list, target_samples: int, class_names: list
) -> tuple:
    """Select tiles to reach target samples while maximizing class balance.

    Arguments:
    ---------
    tile_data: list of tuples
        List of tuples of the form (tile_id, class_counts_dict)
    target_samples: int
        Number of total samples to select.
    class_names: list of str
        List of all possible class names.

    Returns:
    -------
    selected_tiles: list
        List of selected tile ids (indices in `tile_data`)
    final_counts: dict
        Dictionary of class counts for the selected tiles.

    """
    # Convert tile data to a more usable format
    tiles = []
    class_indices = {cls: i for i, cls in enumerate(class_names)}
    num_classes = len(class_names)

    for tile_id, class_counts in tile_data:
        # Create a vector representation of class counts for this tile
        vec = np.zeros(num_classes)
        for cls, count in class_counts.items():
            vec[class_indices[cls]] = count
        tiles.append((tile_id, vec, sum(class_counts.values())))

    # Initialize variables
    selected_tiles = []
    selected_counts = np.zeros(num_classes)
    remaining_tiles = tiles.copy()

    # While we haven't reached target and tiles remain
    while np.sum(selected_counts) < target_samples and remaining_tiles:
        # Calculate current balance ratio (we want to maximize the minimum class)
        current_min = np.min(selected_counts) if np.any(selected_counts) else 0

        # For each remaining tile, calculate how it would improve balance
        scores = []
        for tile_id, vec, total in remaining_tiles:
            new_counts = selected_counts + vec
            new_min = np.min(new_counts)

            # Score based on:
            # 1. How much it improves the minimum class
            # 2. How much it gets us closer to target
            # 3. Prefer tiles that don't overshoot target too much
            remaining_needed = target_samples - np.sum(selected_counts)
            overshoot = max(0, np.sum(new_counts) - target_samples)

            score = (
                new_min - current_min,  # Primary: improve balance
                -overshoot,  # Secondary: minimize overshoot
                min(total, remaining_needed),
            )  # Tertiary: maximize usefulness

            scores.append((score, tile_id, vec, total))

        # Select the tile with the best score
        scores.sort(reverse=True)
        best_score, best_id, best_vec, best_total = scores[0]

        # Add to selection
        selected_tiles.append(best_id)
        selected_counts += best_vec

        # Remove from remaining tiles
        remaining_tiles = [c for c in remaining_tiles if c[0] != best_id]

    # Convert counts back to dictionary
    final_counts = {cls: selected_counts[class_indices[cls]] for cls in class_names}

    return selected_tiles, final_counts
