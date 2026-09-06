# Counting vessels from a segmentation mask, with methods that can split
# touching vessels.
#
# The problem
# Plain connected components counts a group of touching vessels as one object. At
# 256x256 the cell walls between adjacent metaxylem vessels are only a pixel or
# two wide, so they routinely merge in the predicted mask - which is why the
# earlier reconstruction test found ~6.4 components per image while the detector
# was proposing ~16 instances. Any count built on plain components is therefore
# biased low, and biased low by an amount that depends on how crowded the stele
# is, which is exactly the quantity being measured.
#
# Methods
# connected_components
#     Baseline. Fast, but merges anything that touches.
#
# opening Binary opening (erosion then dilation) with a small disk first, to
# break one-pixel bridges, then components. Cheap and helps with thin
# connections, but erosion also shrinks genuinely small vessels and can delete
# them outright, so it trades one bias for another.
#
# watershed Distance transform of the mask, local maxima as seeds, watershed on
# the inverted distance. This is the standard approach for splitting touching
# round objects and is the one that actually separates vessels rather than
# eroding them apart. The parameter that matters is min_distance: too small
# over-segments (one vessel becomes several), too large under-segments (merges
# survive). It is worth choosing that number against ground truth rather than
# by eye - validate_vessel_counting.py does exactly that.
#
# All methods return the same triple so they are interchangeable at the call
# site: (count, labelled image, list of per-vessel areas).

import numpy as np
from scipy import ndimage


def _filter_labels(labelled, min_area):
    # Drops components below min_area and relabels 1..n.
    if labelled.max() == 0:
        return labelled, 0, []

    areas = ndimage.sum(labelled > 0, labelled, range(1, labelled.max() + 1))
    keep = np.where(areas >= min_area)[0] + 1

    out = np.zeros_like(labelled)
    kept_areas = []
    for new_id, old_id in enumerate(keep, start=1):
        out[labelled == old_id] = new_id
        kept_areas.append(float(areas[old_id - 1]))
    return out, len(keep), kept_areas


def count_connected_components(mask, min_area=4, connectivity=2):
    structure = ndimage.generate_binary_structure(2, connectivity)
    labelled, _ = ndimage.label(mask, structure=structure)
    return _filter_labels(labelled, min_area)


def count_with_opening(mask, min_area=4, connectivity=2, radius=1):
    # Breaks thin bridges with a binary opening, then counts components.
    #
    # The opening is applied only to decide the split; areas are then measured by
    # dilating each label back over the ORIGINAL mask, so vessels are not reported
    # as smaller than they really are just because erosion was used to separate
    # them.
    from skimage.morphology import binary_opening, disk

    opened = binary_opening(mask, disk(radius))
    structure = ndimage.generate_binary_structure(2, connectivity)
    labelled, n = ndimage.label(opened, structure=structure)
    if n == 0:
        return labelled, 0, []

    # Give the eroded pixels back to their nearest label, within the original
    # mask, so the split is preserved but the areas stay honest.
    _, nearest = ndimage.distance_transform_edt(labelled == 0, return_indices=True)
    grown = labelled[tuple(nearest)]
    grown[~mask] = 0
    return _filter_labels(grown, min_area)


def count_with_watershed(mask, min_area=4, min_distance=3, connectivity=2):
    # Splits touching vessels using a distance-transform watershed.
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    if not mask.any():
        return np.zeros_like(mask, dtype=int), 0, []

    distance = ndimage.distance_transform_edt(mask)
    coords = peak_local_max(distance, min_distance=min_distance, labels=mask)

    if len(coords) == 0:
        # No interior maximum found (very thin mask) - nothing to split.
        return count_connected_components(mask, min_area, connectivity)

    markers = np.zeros(distance.shape, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)

    labelled = watershed(-distance, markers, mask=mask)
    return _filter_labels(labelled, min_area)


COUNTERS = {
    'connected_components': count_connected_components,
    'opening': count_with_opening,
    'watershed': count_with_watershed,
}


def count_vessels(mask, method='watershed', min_area=4, **kwargs):
    # Dispatches to one of the counting methods.
    #
    # Returns (labelled, count, areas). Unknown keyword arguments are passed
    # through, so watershed's min_distance and opening's radius can be tuned by
    # the caller without changing this signature.
    if method not in COUNTERS:
        raise ValueError(f"unknown method '{method}'. "
                         f"Options: {sorted(COUNTERS)}")

    fn = COUNTERS[method]
    if method == 'watershed':
        kwargs.pop('radius', None)
    elif method == 'opening':
        kwargs.pop('min_distance', None)
    else:
        kwargs.pop('radius', None)
        kwargs.pop('min_distance', None)

    return fn(mask, min_area=min_area, **kwargs)
