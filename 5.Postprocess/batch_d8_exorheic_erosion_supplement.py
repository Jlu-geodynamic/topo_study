#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch D8 Drainage Classification and Erosion Statistics for 2D Landscape Models
===============================================================================

This script performs post-processing on a series of 2D topographic snapshots
produced by a numerical landscape evolution model (e.g., ASPECT). For each
time frame, it:

  1. Reconstructs a regular 2D grid from scattered point data.
  2. Applies light preprocessing: a 3x3 median filter to remove isolated
     spikes, followed by a limited depression-filling step (Priority-Flood,
     Wang & Liu 2006) that only fills closed basins shallower than a user
     depth threshold. Deeper basins are preserved as true endorheic sinks.
  3. Identifies the ocean mask as low-elevation cells (z <= SEA) that are
     8-connected to the left or right lateral model boundary.
  4. Computes a single-flow-direction D8 receiver field for all land cells.
  5. Traces each land cell downstream to classify it as exorheic (drains
     to the ocean boundary), endorheic (terminates in a closed basin), or
     unknown (unresolved / periodic-edge ambiguity).
  6. Aggregates erosion rates and volumes over all time frames to produce
     domain-scale denudation statistics, including cumulative erosion
     volume, area-normalized rates, and five pooled rate metrics.

Classification rules
---------------------
* Ocean  : z <= SEA and 8-connected to the x = left or x = right boundary.
* Exorheic : land cell whose downstream D8 path reaches the ocean boundary.
* Endorheic: land cell whose downstream path terminates in a closed sink.
* Closed depressions shallower than MAX_DEPRESSION_DEPTH are artificially
  filled so that their drainage network connects to the outlet (treated as
  exorheic). Deeper depressions are retained as endorheic basins.

Erosion statistics
------------------
Only cells with 0 < erosion_rate < 10 m/yr are counted. Metrics include
cumulative erosion volume, active erosion area, maximum exterior sediment
flux, normalized erosion depth, denudation rate, and five pooled
time-weighted rate averages (domain-mean, active-mean, exorheic
contribution, exorheic-mean, exorheic-active-mean).

Outputs (written to OUT_DIR)
----------------------------
{case}_exterior_summary_D8.csv   Per-model full statistics table.
all_models_D8_summary.csv       Horizontal cross-model summary (Statistic rows
                                 x model columns).
all_models_D8_A_C.csv          Per-model pooled domain-mean (A), exorheic
                                 contribution (C), and related rate metrics.

References
----------
* O'Callaghan & Mark (1984), Water Resources Research — D8 algorithm.
* Wang & Liu (2006), Computers & Geosciences — Priority-Flood depression
  filling.
* Tucker & Hancock (2010), Geomorphology — drainage area and erosion
  statistics conventions.

Dependencies
------------
numpy, pandas, numba, scipy, pyarrow (optional, for fast CSV reading).

Author  : Wang Yushun
Created : 2026
"""

import os
import glob
import traceback
from collections import deque

import numpy as np
import pandas as pd
import numba
from scipy import ndimage

# =============================================================================
# ============================  User Configuration  ==========================
# =============================================================================

#: Root directory containing all *-topo.csv model output files.
ROOT = r'F:\Cs_Draw'

#: File-name glob pattern used when CASES is empty.
FILE_PATTERN = '*-topo.csv'

#: Explicit list of case basenames (without .csv). If non-empty, only these
#: files are processed; otherwise the script globs all matches of FILE_PATTERN.
CASES = [
    '0407wish1-topo', '0407wish4-topo', '0407wish7-topo', '0407wish10-topo',
    '0407wish13-topo', '0407wish16-topo', '0407wish19-topo', '0407wish22-topo',
    '0407wish25-topo',
    '0420wish1-topo', '0420wish4-topo', '0420wish7-topo', '0420wish10-topo',
    '0420wish13-topo', '0420wish16-topo', '0420wish19-topo', '0420wish22-topo',
    '0420wish25-topo',
    '0515wish1-topo', '0515wish4-topo', '0515wish7-topo', '0515wish10-topo',
    '0515wish13-topo', '0515wish16-topo', '0515wish19-topo', '0515wish22-topo',
    '0515wish25-topo',
]

#: Directory where all result CSV files are written.
OUT_DIR = os.path.join(ROOT, 'd8_batch_results')

#: Sea-level elevation (m). Cells with z <= SEA that connect to the lateral
#: boundary are classified as ocean.
SEA = 0.0

#: Conversion factor between model output time index and physical time (Myr).
#: Time index 1 -> 0.1 Ma, Time index 200 -> 20 Ma.
TIME_STEP_MA = 0.1

#: Y-boundary condition along the strike direction.
#:   'periodic' — periodic wrap-around (e.g., along-strike symmetry).
#:   'closed'   — no flow across y boundaries; edges act as no-flow walls.
#:   'unknown'  — y-edge cells are flagged UNKNOWN rather than forced to outlet.
Y_MODE = 'periodic'

#: Vertical tolerance (m) used to detect flat / near-flat terrain.
FLAT_TOL = 1.0e-6

#: Chunk size (in rows) for streaming CSV reading.
CSV_CHUNKSIZE = 1_000_000

#: CSV reader engine. 'pyarrow' is faster; falls back to 'c' on failure.
CSV_ENGINE = 'pyarrow'

# -----------------------------------------------------------------------------
# Preprocessing switches
# -----------------------------------------------------------------------------

#: If True, apply a 3x3 median filter to suppress isolated spike cells before
#: flow routing. Local-minimum (depression-center) cells are protected.
SMOOTH_ENABLED = True

#: Median-filter window size (cells). Use 3 or 5.
SMOOTH_SIZE = 3

#: Maximum closed-depression depth (m) that is artificially filled by
#: Priority-Flood. Depressions deeper than this are preserved as true
#: endorheic basins. Set to np.inf to fill every depression, or 0 to skip
#: filling entirely (strict D8 with all sinks retained).
MAX_DEPRESSION_DEPTH = 200.0

# =============================================================================
# ============================  Receiver Codes  ===============================
# =============================================================================
# Non-negative values encode the flattened downstream cell index.
# Negative values are sentinel codes.

RECV_NODATA = -9    #: Cell is outside the valid domain.
RECV_UNKNOWN = -4   #: Unresolved at an ambiguous boundary (Y_MODE='unknown').
RECV_UNSET = -3     #: Land cell whose receiver has not been computed yet.
RECV_OUTLET = -2    #: Cell drains directly to the ocean / lateral outlet.
RECV_SINK = -1      #: Cell is the bottom of a closed endorheic basin.

# =============================================================================
# ============================  Class Codes  ==================================
# =============================================================================

CLS_NODATA = -1     #: Outside valid domain.
CLS_OCEAN = 0       #: Ocean cell.
CLS_EXORHEIC = 1    #: Exorheic land (drains to ocean).
CLS_ENDORHEIC = 2   #: Endorheic land (closed basin).
CLS_UNKNOWN = 3     #: Unresolved / ambiguous classification.

#: Temporary sentinel used during classification (land not yet traced).
CLS_UNRESOLVED = -2


# =============================================================================
# ======================  Streaming CSV Reader  ===============================
# =============================================================================

def _iter_raw_chunks(path, usecols, chunksize):
    """Yield raw pandas DataFrames of *usecols* from *path* in row chunks.

    Tries the pyarrow streaming reader first for speed; on any exception
    falls back to the pandas C-engine chunked reader.
    """
    if CSV_ENGINE == 'pyarrow':
        try:
            import pyarrow.csv as pacsv
            # Approximate ~64 bytes per row to size the read block.
            read_opts = pacsv.ReadOptions(block_size=chunksize * 64)
            conv_opts = pacsv.ConvertOptions(include_columns=usecols)
            reader = pacsv.open_csv(
                path, read_options=read_opts, convert_options=conv_opts
            )
            for record_batch in reader:
                yield record_batch.to_pandas()
            return
        except Exception as exc:
            print(f'pyarrow reader failed, falling back to C engine: {exc}')

    for chunk in pd.read_csv(
        path, usecols=usecols, chunksize=chunksize, engine='c'
    ):
        yield chunk


def iter_time_groups(path, usecols, chunksize=CSV_CHUNKSIZE):
    """Yield (time_value, full_DataFrame) pairs, one per Time group.

    The source CSV must be sorted by Time in non-decreasing order.
    This function streams through the file, accumulating all rows that share
    the same Time value into a single DataFrame before yielding.
    """
    current_t = None
    parts = []
    last_seen_t = None

    for chunk in _iter_raw_chunks(path, usecols, chunksize):
        tvals = chunk['Time'].to_numpy()
        if tvals.size and np.any(tvals[1:] < tvals[:-1]):
            raise ValueError(
                'Time is not sorted within a chunk; cannot stream.'
            )

        for t, g in chunk.groupby('Time', sort=False):
            t = float(t)
            if last_seen_t is not None and t < last_seen_t:
                raise ValueError(
                    'Time decreases between chunks; sort the source file '
                    'by Time first.'
                )
            last_seen_t = t

            if current_t is None:
                current_t = t
                parts = [g]
            elif t == current_t:
                parts.append(g)
            else:
                yield current_t, pd.concat(parts, ignore_index=True)
                current_t = t
                parts = [g]

    if current_t is None:
        return
    yield current_t, pd.concat(parts, ignore_index=True)


# =============================================================================
# ======================  Grid Geometry & Neighbors  ==========================
# =============================================================================

def grid_spacing(xs, ys):
    """Return (dx, dy) cell spacing in meters, inferred from coordinate arrays."""
    dx = float((xs[-1] - xs[0]) / (len(xs) - 1)) if len(xs) > 1 else 1.0
    dy = float((ys[-1] - ys[0]) / (len(ys) - 1)) if len(ys) > 1 else dx
    if not np.isfinite(dx) or dx <= 0 or not np.isfinite(dy) or dy <= 0:
        raise ValueError('Coordinates must be strictly increasing.')
    return dx, dy


def neighbors(i, j, ny, nx, dx, dy):
    """Yield (ii, jj, distance) for all 8 D8 neighbors of cell (i, j).

    The x direction is always closed (no wrap). The y direction wraps
    periodically when Y_MODE == 'periodic'.
    """
    dis = []
    for di in (-1, 0, 1):
        if ny == 1 and di != 0:
            continue
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            jj = j + dj
            if jj < 0 or jj >= nx:
                continue

            raw_i = i + di
            if 0 <= raw_i < ny:
                ii = raw_i
            elif Y_MODE == 'periodic' and ny > 1:
                ii = raw_i % ny
            else:
                continue

            if ii == i and jj == j:
                continue
            dist = float(np.hypot(abs(dj) * dx, abs(di) * dy))
            dis.append((ii, jj, dist))

    # When ny == 2, periodic wrapping can produce the same neighbor twice;
    # keep the entry with the shorter distance.
    best = {}
    for ii, jj, d in dis:
        key = (ii, jj)
        if key not in best or d < best[key]:
            best[key] = d
    for (ii, jj), d in best.items():
        yield ii, jj, d


# =============================================================================
# ======================  Ocean Identification  ===============================
# =============================================================================

def edge_connected_ocean(z, xs, ys):
    """Return a boolean mask of ocean cells.

    A cell is ocean if:
      * it is below or at sea level (z <= SEA), and
      * it is 8-connected to the left or right lateral boundary.

    Only the x = 0 and x = nx-1 edges are considered open; inland below-sea
    depressions are NOT counted as ocean. When Y_MODE is periodic, the mask
    is padded by one row on top and bottom before connected-component
    labeling to correctly handle along-strike wrap.
    """
    valid = np.isfinite(z)
    low = valid & (z <= SEA)
    if not np.any(low):
        return np.zeros(z.shape, dtype=bool)

    conn = np.ones((3, 3), dtype=int)  # 8-connectivity
    if Y_MODE == 'periodic' and z.shape[0] > 1:
        padded = np.vstack([low[-1:], low, low[:1]])
        labels, _ = ndimage.label(padded, structure=conn)
        labels = labels[1:-1, :]
    else:
        labels, _ = ndimage.label(low, structure=conn)

    edge_labels = np.unique(np.concatenate([labels[:, 0], labels[:, -1]]))
    edge_labels = edge_labels[edge_labels > 0]
    return np.isin(labels, edge_labels) & low


# =============================================================================
# ======================  Depression Filling (Priority-Flood)  ===============
# =============================================================================

@numba.njit(cache=False)
def _priority_flood_full(z, edge_mask, valid, y_periodic):
    """Run the full Priority-Flood algorithm (Wang & Liu, 2006).

    Fills every closed depression that can flow to a known water level
    (the *edge_mask* cells). Each edge cell starts in the priority queue
    at its own elevation. Cells are popped in order of lowest water level;
    any neighbor lower than the current level is raised to that level and
    pushed back into the queue.

    Parameters
    ----------
    z : (ny, nx) float array
        Original topography (m).
    edge_mask : (ny, nx) bool array
        Cells with a known water level (ocean + open boundary cells).
    valid : (ny, nx) bool array
        True where z is finite.
    y_periodic : bool
        Whether to wrap in the y direction.

    Returns
    -------
    filled : (ny, nx) float array
        Depression-filled topography.
    """
    ny, nx = z.shape
    filled = z.copy()
    closed = edge_mask.copy()

    n_total = ny * nx
    heap_h = np.empty(n_total, dtype=np.float64)
    heap_idx = np.empty(n_total, dtype=np.int64)
    heap_size = 0

    # Initialize: push all edge cells into the min-heap at their own elevation.
    for i in range(ny):
        for j in range(nx):
            if edge_mask[i, j]:
                heap_h[heap_size] = z[i, j]
                heap_idx[heap_size] = i * nx + j
                pos = heap_size
                heap_size += 1
                # Sift up.
                while pos > 0:
                    parent = (pos - 1) // 2
                    if heap_h[pos] < heap_h[parent]:
                        tmp_h = heap_h[pos]; heap_h[pos] = heap_h[parent]; heap_h[parent] = tmp_h
                        tmp_i = heap_idx[pos]; heap_idx[pos] = heap_idx[parent]; heap_idx[parent] = tmp_i
                        pos = parent
                    else:
                        break

    while heap_size > 0:
        # Pop the minimum.
        h = heap_h[0]
        idx = heap_idx[0]
        heap_size -= 1
        if heap_size > 0:
            heap_h[0] = heap_h[heap_size]
            heap_idx[0] = heap_idx[heap_size]
            pos = 0
            # Sift down.
            while True:
                left = 2 * pos + 1
                right = 2 * pos + 2
                smallest = pos
                if left < heap_size and heap_h[left] < heap_h[smallest]:
                    smallest = left
                if right < heap_size and heap_h[right] < heap_h[smallest]:
                    smallest = right
                if smallest == pos:
                    break
                tmp_h = heap_h[pos]; heap_h[pos] = heap_h[smallest]; heap_h[smallest] = tmp_h
                tmp_i = heap_idx[pos]; heap_idx[pos] = heap_idx[smallest]; heap_idx[smallest] = tmp_i
                pos = smallest

        i = idx // nx
        j = idx % nx

        # Visit all 8 neighbors.
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue
                ii = i + di
                jj = j + dj
                # Periodic wrap in y.
                if ii < 0:
                    if y_periodic and ny > 1:
                        ii = ii + ny
                    else:
                        continue
                elif ii >= ny:
                    if y_periodic and ny > 1:
                        ii = ii - ny
                    else:
                        continue
                if jj < 0 or jj >= nx:
                    continue
                if not valid[ii, jj] or closed[ii, jj]:
                    continue
                closed[ii, jj] = True
                zn = z[ii, jj]
                if zn < h:
                    # This cell is inside a depression; raise it to the
                    # current spill level.
                    filled[ii, jj] = h
                    new_h = h
                else:
                    new_h = zn
                # Push into the heap and sift up.
                heap_h[heap_size] = new_h
                heap_idx[heap_size] = ii * nx + jj
                pos = heap_size
                heap_size += 1
                while pos > 0:
                    parent = (pos - 1) // 2
                    if heap_h[pos] < heap_h[parent]:
                        tmp_h = heap_h[pos]; heap_h[pos] = heap_h[parent]; heap_h[parent] = tmp_h
                        tmp_i = heap_idx[pos]; heap_idx[pos] = heap_idx[parent]; heap_idx[parent] = tmp_i
                        pos = parent
                    else:
                        break

    return filled


def fill_depressions(z, ocean, valid, max_depth):
    """Fill only closed depressions shallower than *max_depth* (m).

    Workflow
    ---------
    1. Run a full Priority-Flood to obtain *filled_full* (every depression
       filled to its spill elevation).
    2. Identify connected components of "depression cells" (filled_full > z).
    3. For each component, compute its maximum fill depth. Components with
       max depth <= *max_depth* are adopted from *filled_full*; deeper
       components are left at their original elevation and remain as
       endorheic sinks.

    Parameters
    ----------
    z : (ny, nx) float array
        Original topography.
    ocean : (ny, nx) bool array
        Ocean mask (cells already at base level).
    valid : (ny, nx) bool array
        Finite-elevation mask.
    max_depth : float
        Maximum depression depth (m) to fill. Use np.inf to fill all
        depressions, or 0 to skip filling.

    Returns
    -------
    out : (ny, nx) float array
        Topography with shallow depressions filled, deep ones preserved.
    """
    # No filling requested → return original.
    if max_depth <= 0:
        return z.copy()

    # Build the known-water-level mask: ocean + all open boundary cells.
    ny, nx = z.shape
    edge_mask = ocean.copy()
    edge_mask[:, 0] = edge_mask[:, 0] | valid[:, 0]
    edge_mask[:, -1] = edge_mask[:, -1] | valid[:, -1]
    if Y_MODE != 'periodic' or ny == 1:
        edge_mask[0, :] = edge_mask[0, :] | valid[0, :]
        edge_mask[-1, :] = edge_mask[-1, :] | valid[-1, :]

    y_periodic = (Y_MODE == 'periodic' and ny > 1)
    filled_full = _priority_flood_full(z, edge_mask, valid, y_periodic)

    # Infinite threshold → fill everything.
    if not np.isfinite(max_depth):
        return filled_full

    # Find connected components of depressions.
    depth = filled_full - z
    depression = (depth > 1e-9) & valid
    if not depression.any():
        return z.copy()

    conn = np.ones((3, 3), dtype=int)  # 8-connectivity
    if y_periodic:
        padded = np.vstack([depression[-1:], depression, depression[:1]])
        labels, n = ndimage.label(padded, structure=conn)
        labels = labels[1:-1, :]
    else:
        labels, n = ndimage.label(depression, structure=conn)

    output = z.copy()
    flat_labels = labels.ravel()
    flat_depth = depth.ravel()
    for lab in range(1, n + 1):
        mask = flat_labels == lab
        if not mask.any():
            continue
        max_d = float(flat_depth[mask].max())
        if max_d <= max_depth:
            # Shallow enough → fill this component.
            output_flat = output.ravel()
            filled_flat = filled_full.ravel()
            for k in np.where(mask)[0]:
                output_flat[k] = filled_flat[k]
    return output


# =============================================================================
# ======================  Terrain Smoothing  ===================================
# =============================================================================

def smooth_terrain(z, valid, size=3):
    """Apply a median filter to suppress isolated spike cells.

    Local-minimum cells (depression centers) are protected so that genuine
    closed basins are not flattened by the filter.

    Parameters
    ----------
    z : (ny, nx) float array
        Topography.
    valid : (ny, nx) bool array
        Finite mask.
    size : int
        Filter window size (cells).

    Returns
    -------
    smoothed : (ny, nx) float array
        Smoothed topography, with local minima left unchanged.
    """
    if size <= 1:
        return z.copy()

    # Identify local minima: a cell whose elevation is <= the minimum of
    # its 3x3 neighborhood.
    z_for_min = np.where(valid, z, np.inf)
    neighbor_min = ndimage.minimum_filter(z_for_min, size=size, mode='nearest')
    is_local_min = (z <= neighbor_min + 1e-9) & valid

    # Median filter on valid cells.
    z_work = np.where(valid, z, 0.0)
    smoothed = ndimage.median_filter(z_work, size=size, mode='nearest')

    # Apply smoothing only to non-local-minimum cells.
    return np.where(valid & ~is_local_min, smoothed, z)


# =============================================================================
# ======================  D8 Receiver Routing  =================================
# =============================================================================

def _shift_neighbor(arr, di, dj, fill_value):
    """Return arr shifted by (di, dj), aligned to the original grid.

    The x direction does not wrap; out-of-bounds columns are set to
    *fill_value*. The y direction wraps only when Y_MODE == 'periodic'.
    """
    ny, nx = arr.shape
    out = np.roll(arr, shift=(-di, -dj), axis=(0, 1))

    if dj == 1:
        out[:, -1] = fill_value
    elif dj == -1:
        out[:, 0] = fill_value

    if Y_MODE != 'periodic' or ny == 1:
        if di == 1:
            out[-1, :] = fill_value
        elif di == -1:
            out[0, :] = fill_value
    return out


def build_receivers(z, xs, ys):
    """Compute the full 2D D8 receiver field for land cells.

    The steepest-descent D8 direction is computed vectorially for all
    downslope cells. Flat / near-flat cells and unresolved depressions are
    then handled with a breadth-first flood from existing drainage points.

    Preprocessing (controlled by global configuration):
      1. Median-filter smoothing (SMOOTH_ENABLED) to remove spikes.
      2. Limited depression filling (MAX_DEPRESSION_DEPTH) to connect
         shallow basins to the outlet.

    Returns
    -------
    receiver : (ny, nx) int64 array
        Receiver codes. Non-negative values are flattened downstream cell
        indices; negative values are RECV_* sentinels.
    ocean : (ny, nx) bool array
        Ocean mask.
    terrain : (ny, nx) bool array
        Land mask (valid and not ocean).
    """
    if Y_MODE not in {'periodic', 'closed', 'unknown'}:
        raise ValueError(
            "Y_MODE must be one of 'periodic', 'closed', or 'unknown'."
        )

    ny, nx = z.shape
    dx, dy = grid_spacing(xs, ys)
    valid = np.isfinite(z)

    # Preprocessing step 1: light median smoothing.
    if SMOOTH_ENABLED and SMOOTH_SIZE > 1:
        z = smooth_terrain(z, valid, SMOOTH_SIZE)

    # Ocean mask is computed on the smoothed topography (same terrain used
    # for flow routing).
    ocean = edge_connected_ocean(z, xs, ys)

    # Preprocessing step 2: fill shallow depressions.
    # Ocean cells themselves are never raised, so the ocean mask is already
    # correct after filling; we do not need to recompute it.
    z = fill_depressions(z, ocean, valid, MAX_DEPRESSION_DEPTH)
    terrain = valid & ~ocean

    receiver = np.full(z.shape, RECV_NODATA, dtype=np.int64)
    receiver[terrain] = RECV_UNSET

    best_slope = np.full(z.shape, -np.inf, dtype=float)
    idx = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)

    # --- Vectorized steepest-descent D8 over all 8 neighbor directions ---
    for di in (-1, 0, 1):
        if ny == 1 and di != 0:
            continue
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue

            zn = _shift_neighbor(z, di, dj, np.nan)
            on = _shift_neighbor(ocean, di, dj, False)
            tn = _shift_neighbor(terrain, di, dj, False)
            inbr = _shift_neighbor(idx, di, dj, -1)

            dist = float(np.hypot(abs(dj) * dx, abs(di) * dy))
            drop = z - zn
            # Candidate: land cell, neighbor is land or ocean, neighbor
            # elevation is finite, and there is a meaningful downslope drop.
            cand = terrain & (on | tn) & np.isfinite(zn) & (drop > FLAT_TOL)
            slope = np.where(cand, drop / dist, -np.inf)
            better = slope > best_slope
            if not np.any(better):
                continue

            # If the neighbor is ocean, the cell drains directly to outlet.
            chosen = np.where(on, RECV_OUTLET, inbr)
            receiver[better] = chosen[better]
            best_slope[better] = slope[better]

    # The left and right lateral boundaries are open to the ocean: any land
    # cell on x = 0 or x = nx-1 is treated as an outlet regardless of its
    # immediate neighbor (open-boundary condition).
    receiver[terrain[:, 0], 0] = RECV_OUTLET
    receiver[terrain[:, -1], -1] = RECV_OUTLET

    # If Y_MODE is 'unknown', flag y-edge land cells as UNKNOWN instead of
    # forcing them to an outlet.
    if Y_MODE == 'unknown' and ny > 1:
        yedge = np.zeros_like(terrain)
        yedge[0, :] = terrain[0, :]
        yedge[-1, :] = terrain[-1, :]
        yedge[:, 0] = False
        yedge[:, -1] = False
        receiver[yedge] = RECV_UNKNOWN

    # --- Resolve flat / unresolved cells via BFS flood ---
    unresolved_mask = terrain & (receiver == RECV_UNSET)
    visited = np.zeros(z.shape, dtype=bool)

    for si, sj in zip(*np.where(unresolved_mask)):
        if visited[si, sj] or receiver[si, sj] != RECV_UNSET:
            continue

        h0 = float(z[si, sj])
        comp = []
        q = deque([(int(si), int(sj))])
        visited[si, sj] = True

        # Grow the connected component of near-equal-elevation cells.
        while q:
            i, j = q.popleft()
            comp.append((i, j))
            h = float(z[i, j])
            for ii, jj, _ in neighbors(i, j, ny, nx, dx, dy):
                if not terrain[ii, jj]:
                    continue
                hh = float(z[ii, jj])
                if abs(hh - h) <= FLAT_TOL and abs(hh - h0) <= FLAT_TOL:
                    if not visited[ii, jj]:
                        visited[ii, jj] = True
                        q.append((ii, jj))

        unresolved = [(i, j) for i, j in comp if receiver[i, j] == RECV_UNSET]
        if not unresolved:
            continue

        # Cells in the component that already have a receiver act as drains.
        drains = [(i, j) for i, j in comp if receiver[i, j] != RECV_UNSET]
        if not drains:
            # No drain in the component → closed endorheic sink.
            for i, j in comp:
                receiver[i, j] = RECV_SINK
            continue

        # Flood inward from the drain cells: each unresolved cell receives
        # as its receiver the upstream-adjacent cell on the BFS frontier.
        comp_set = set(comp)
        flat_seen = set(drains)
        fq = deque(drains)
        while fq:
            i, j = fq.popleft()
            for ii, jj, _ in neighbors(i, j, ny, nx, dx, dy):
                if (ii, jj) not in comp_set or (ii, jj) in flat_seen:
                    continue
                flat_seen.add((ii, jj))
                if receiver[ii, jj] == RECV_UNSET:
                    receiver[ii, j] = i * nx + j
                fq.append((ii, jj))

        # Any remaining unresolved cell in the component is a sink.
        for i, j in unresolved:
            if receiver[i, j] == RECV_UNSET:
                receiver[i, j] = RECV_SINK

    return receiver, ocean, terrain


# =============================================================================
# ======================  Drainage Classification  =============================
# =============================================================================

def classify_from_receivers(receiver, ocean, terrain):
    """Classify every land cell as exorheic, endorheic, or unknown.

    Starting from each unclassified land cell, walk downstream along the
    receiver chain until:
      * a cell already classified is found → inherit its class,
      * an outlet (RECV_OUTLET) is reached → exorheic,
      * a sink (RECV_SINK) is reached → endorheic,
      * a sentinel / cycle is encountered → unknown.

    All cells along the traced path are then labeled with the result,
    giving O(N) amortized runtime across the grid.
    """
    ny, nx = receiver.shape
    cls = np.full(receiver.shape, CLS_NODATA, dtype=np.int8)
    cls[ocean] = CLS_OCEAN
    cls[terrain] = CLS_UNRESOLVED  # temporary: land not yet traced

    for si, sj in zip(*np.where(terrain)):
        if cls[si, sj] in (CLS_EXORHEIC, CLS_ENDORHEIC, CLS_UNKNOWN):
            continue

        path = []
        seen = set()
        cur = int(si) * nx + int(sj)

        while True:
            i, j = divmod(cur, nx)
            if cls[i, j] in (CLS_EXORHEIC, CLS_ENDORHEIC, CLS_UNKNOWN):
                result = int(cls[i, j])
                break
            if cur in seen:
                # Cycle detected in the receiver chain.
                result = CLS_UNKNOWN
                break

            seen.add(cur)
            path.append(cur)
            rec = int(receiver[i, j])
            if rec == RECV_OUTLET:
                result = CLS_EXORHEIC
                break
            if rec == RECV_SINK:
                result = CLS_ENDORHEIC
                break
            if rec in (RECV_UNKNOWN, RECV_NODATA, RECV_UNSET):
                result = CLS_UNKNOWN
                break
            cur = rec

        # Label all cells along the traced path with the result.
        for cidx in path:
            i, j = divmod(cidx, nx)
            cls[i, j] = result

    return cls


# =============================================================================
# ======================  Grid Reconstruction  ================================
# =============================================================================

def make_grid(group, xs=None, ys=None):
    """Rasterize a scattered point DataFrame onto a regular 2D grid.

    Uses np.searchsorted for fast coordinate → index mapping.
    """
    if xs is None:
        xs = np.sort(group['Points:0'].dropna().unique()).astype(float)
    if ys is None:
        ys = np.sort(group['Points:1'].dropna().unique()).astype(float)

    xv = group['Points:0'].to_numpy(float)
    yv = group['Points:1'].to_numpy(float)
    xi = np.searchsorted(xs, xv)
    yi = np.searchsorted(ys, yv)

    in_grid = (
        (xi >= 0) & (xi < len(xs)) & (yi >= 0) & (yi < len(ys)) &
        np.isclose(xs[np.clip(xi, 0, len(xs) - 1)], xv) &
        np.isclose(ys[np.clip(yi, 0, len(ys) - 1)], yv)
    )
    if not np.all(in_grid):
        raise ValueError('Coordinate grid changes between time frames.')

    out = {}
    for col in group.columns:
        if col in {'Time', 'Points:0', 'Points:1'}:
            continue
        arr = np.full((len(ys), len(xs)), np.nan, dtype=float)
        arr[yi, xi] = group[col].to_numpy(float)
        out[col] = arr
    return xs, ys, out


# =============================================================================
# ======================  Per-File Processing  =================================
# =============================================================================

def process_file(path):
    """Stream-process one model CSV file and return its statistics.

    Returns
    -------
    case : str
        File basename without .csv.
    result_df : pd.DataFrame
        Two-column table (Statistic, Value) with all summary metrics.
    metrics : dict
        Key metrics used for the cross-model A/C summary.
    """
    case = os.path.basename(path)[:-4]  # strip .csv

    # --- Accumulators across all time frames ---
    cum_vol_ext = 0.0
    cum_vol_int = 0.0
    cum_vol_tot = 0.0

    sum_ext_rate_dt = 0.0
    sum_all_rate_dt = 0.0
    cnt_ext_dt = 0.0
    cnt_all_dt = 0.0
    cnt_active_ext_dt = 0.0
    cnt_active_all_dt = 0.0

    total_span_ma = 0.0
    cells_per_frame = 0
    dx = dy = None

    max_flux = 0.0
    max_flux_t_ma = 0.0
    max_int_flux = 0.0
    max_tot_flux = 0.0

    # Active erosion area on the last frame only.
    last_active_all = 0
    last_active_ext = 0

    xs = ys = None
    prev_t = None
    frame_no = 0

    usecols = ['Time', 'Points:0', 'Points:1', 'topography', 'erosion_rate']
    for t_step, group in iter_time_groups(path, usecols):
        xs, ys, grids = make_grid(group, xs, ys)
        z = grids['topography']
        erosion = grids['erosion_rate']
        ny, nx = z.shape

        if dx is None:
            dx = float(np.median(np.diff(xs))) if nx > 1 else 1.0
            dy = float(np.median(np.diff(ys))) if ny > 1 else dx

        # Physical time step (Myr). First frame uses the nominal step.
        if frame_no == 0:
            dt_ma = TIME_STEP_MA
        else:
            dt_ma = (t_step - prev_t) * TIME_STEP_MA
        prev_t = t_step
        frame_no += 1

        # D8 flow routing and drainage classification.
        receiver, ocean, terrain = build_receivers(z, xs, ys)
        cls = classify_from_receivers(receiver, ocean, terrain)

        ext_mask = cls == CLS_EXORHEIC
        int_mask = cls == CLS_ENDORHEIC

        # Only accept physically reasonable erosion rates (m/yr).
        valid_erosion = np.where(
            (erosion > 0) & (erosion < 10), erosion, 0.0
        )
        erosion_mask = valid_erosion > 0

        # Volume per cell in this frame: erosion_rate (m/yr) * cell_area (m²)
        # * dt (yr) = m³.
        cell_volume = dx * dy * dt_ma * 1e6
        vol_ext = float(np.sum(valid_erosion[ext_mask])) * cell_volume
        vol_int = float(np.sum(valid_erosion[int_mask])) * cell_volume
        vol_tot = float(np.sum(valid_erosion)) * cell_volume

        active_all = int(np.sum(erosion_mask))
        active_ext = int(np.sum(erosion_mask & ext_mask))
        ext_rate_sum = float(np.sum(valid_erosion[ext_mask]))
        all_rate_sum = float(np.sum(valid_erosion))

        cum_vol_ext += vol_ext
        cum_vol_int += vol_int
        cum_vol_tot += vol_tot

        # Time-weighted rate accumulators.
        sum_ext_rate_dt += ext_rate_sum * dt_ma
        sum_all_rate_dt += all_rate_sum * dt_ma
        cnt_ext_dt += float(ext_mask.sum()) * dt_ma
        cnt_all_dt += float(nx * ny) * dt_ma
        cnt_active_ext_dt += float(active_ext) * dt_ma
        cnt_active_all_dt += float(active_all) * dt_ma

        total_span_ma += dt_ma
        cells_per_frame = nx * ny

        last_active_all = active_all
        last_active_ext = active_ext

        # Track the maximum exorheic sediment flux.
        dt_yr = dt_ma * 1e6
        if dt_yr > 0:
            flux_ext = vol_ext / dt_yr
            if flux_ext > max_flux:
                max_flux = flux_ext
                max_flux_t_ma = t_step * TIME_STEP_MA
                max_int_flux = vol_int / dt_yr
                max_tot_flux = vol_tot / dt_yr

    if frame_no == 0:
        raise ValueError('No time frames found in file.')

    cell_area = dx * dy
    total_model_area_m2 = cells_per_frame * cell_area
    total_model_area_km2 = total_model_area_m2 / 1e6
    total_span_yr = total_span_ma * 1e6

    # Last-frame active erosion areas.
    total_erosion_area_km2 = last_active_all * cell_area / 1e6
    ext_erosion_area_km2 = last_active_ext * cell_area / 1e6
    ext_area_ratio = (
        last_active_ext / last_active_all * 100 if last_active_all else 0.0
    )

    # Average exterior erosion rate: cumulative exterior volume divided by
    # last-frame exterior area and total time span (mm/yr).
    avg_ext_rate = 0.0
    last_ext_area_m2 = ext_erosion_area_km2 * 1e6
    if last_ext_area_m2 > 0 and total_span_yr > 0:
        avg_ext_rate = cum_vol_ext / last_ext_area_m2 / total_span_yr * 1000

    # Normalized erosion depth (m), spread over the whole model area.
    norm_total_m = (
        cum_vol_tot / total_model_area_m2 if total_model_area_m2 else 0.0
    )
    norm_ext_m = (
        cum_vol_ext / total_model_area_m2 if total_model_area_m2 else 0.0
    )

    # Denudation rate (mm/yr).
    denud_total = (
        norm_total_m / total_span_yr * 1000 if total_span_yr else 0.0
    )
    denud_ext = (
        norm_ext_m / total_span_yr * 1000 if total_span_yr else 0.0
    )

    # Five pooled time-weighted rate averages (mm/yr):
    #   A — domain mean (all cells, including inactive / ocean)
    #   B — mean over active-erosion cells only
    #   C — exorheic contribution averaged over the whole domain
    #   D — mean over exorheic land cells
    #   E — mean over active exorheic cells only
    pooled_A = sum_all_rate_dt / cnt_all_dt * 1000 if cnt_all_dt else 0.0
    pooled_B = (
        sum_all_rate_dt / cnt_active_all_dt * 1000
        if cnt_active_all_dt else 0.0
    )
    pooled_C = sum_ext_rate_dt / cnt_all_dt * 1000 if cnt_all_dt else 0.0
    pooled_D = sum_ext_rate_dt / cnt_ext_dt * 1000 if cnt_ext_dt else 0.0
    pooled_E = (
        sum_ext_rate_dt / cnt_active_ext_dt * 1000
        if cnt_active_ext_dt else 0.0
    )

    ext_percentage = cum_vol_ext / cum_vol_tot * 100 if cum_vol_tot else 0.0

    result_df = pd.DataFrame({
        'Statistic': [
            'Cumulative_Exterior_Erosion_Billion_m3',
            'Cumulative_Total_Erosion_Billion_m3',
            'Exterior_Percentage',
            'Total_Erosion_Area_km2',
            'Exterior_Erosion_Area_km2',
            'Exterior_Erosion_Area_Percentage',
            'Average_Exterior_Erosion_Rate_mm_per_yr',
            'Max_Exterior_Flux_Thousand_m3_per_yr',
            'Max_Flux_Time_Ma',
            'Max_Flux_Corresponding_Interior_Flux_Thousand_m3_per_yr',
            'Max_Flux_Corresponding_Total_Flux_Thousand_m3_per_yr',
            'Normalized_Total_Erosion_m',
            'Normalized_Exterior_Erosion_m',
            'Total_Model_Area_km2',
            'Total_Time_Span_Ma',
            'Denudation_Rate_Total_mm_per_yr',
            'Denudation_Rate_Exterior_mm_per_yr',
            'Pooled_Domain_Mean_Rate_mm_per_yr',
            'Pooled_ActiveErosion_Mean_Rate_mm_per_yr',
            'Pooled_Exorheic_Contribution_mm_per_yr',
            'Pooled_Exorheic_Mean_Rate_mm_per_yr',
            'Pooled_Exorheic_Active_Mean_Rate_mm_per_yr',
        ],
        'Value': [
            cum_vol_ext / 1e9,
            cum_vol_tot / 1e9,
            ext_percentage,
            total_erosion_area_km2,
            ext_erosion_area_km2,
            ext_area_ratio,
            avg_ext_rate,
            max_flux / 1e3,
            max_flux_t_ma,
            max_int_flux / 1e3,
            max_tot_flux / 1e3,
            norm_total_m,
            norm_ext_m,
            total_model_area_km2,
            total_span_ma,
            denud_total,
            denud_ext,
            pooled_A,
            pooled_B,
            pooled_C,
            pooled_D,
            pooled_E,
        ]
    })

    metrics = {
        'case': case,
        'frames': frame_no,
        'A': pooled_A,
        'C': pooled_C,
        'B': pooled_B,
        'D': pooled_D,
        'E': pooled_E,
        'cum_total_billion_m3': cum_vol_tot / 1e9,
        'cum_ext_billion_m3': cum_vol_ext / 1e9,
        'ext_percentage': ext_percentage,
        'total_span_ma': total_span_ma,
        'model_area_km2': total_model_area_km2,
        'max_flux_thousand_m3_yr': max_flux / 1e3,
    }
    return case, result_df, metrics


# =============================================================================
# ======================  Main Entry Point  ===================================
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Determine which files to process.
    if CASES:
        files = sorted(os.path.join(ROOT, c + '.csv') for c in CASES)
        files = [f for f in files if os.path.isfile(f)]
    else:
        files = sorted(glob.glob(os.path.join(ROOT, FILE_PATTERN)))

    if not files:
        print(f'No files matched: {os.path.join(ROOT, FILE_PATTERN)}')
        return

    print(f'Found {len(files)} model files. Starting batch processing...')

    all_rows = {}     # {case: Series(Statistic -> Value)}
    ac_rows = []      # per-model A/C summary rows
    ok = 0

    for k, path in enumerate(files, 1):
        case = os.path.basename(path)
        print(f'\n[{k}/{len(files)}] {case}')
        try:
            name, result_df, metrics = process_file(path)
        except Exception as exc:
            print(f'  FAILED: {exc}')
            traceback.print_exc()
            continue

        # Write per-model summary.
        out_path = os.path.join(OUT_DIR, f'{name}_exterior_summary_D8.csv')
        result_df.to_csv(out_path, index=False, encoding='utf-8-sig')

        all_rows[name] = result_df.set_index('Statistic')['Value']
        ac_rows.append({
            'Model': name,
            'Pooled_Domain_Mean_Rate_A_mm_per_yr': metrics['A'],
            'Pooled_Exorheic_Contribution_C_mm_per_yr': metrics['C'],
            'C_over_A_Percent': (
                metrics['C'] / metrics['A'] * 100 if metrics['A'] else 0.0
            ),
            'Pooled_ActiveErosion_B_mm_per_yr': metrics['B'],
            'Pooled_Exorheic_Mean_D_mm_per_yr': metrics['D'],
            'Pooled_Exorheic_Active_E_mm_per_yr': metrics['E'],
            'Cumulative_Total_Billion_m3': metrics['cum_total_billion_m3'],
            'Cumulative_Exterior_Billion_m3': metrics['cum_ext_billion_m3'],
            'Exterior_Percentage': metrics['ext_percentage'],
            'Total_Time_Span_Ma': metrics['total_span_ma'],
            'Total_Model_Area_km2': metrics['model_area_km2'],
            'Max_Exterior_Flux_Thousand_m3_per_yr':
                metrics['max_flux_thousand_m3_yr'],
        })

        print(f"  OK  A={metrics['A']:.4f}  C={metrics['C']:.4f} mm/yr  "
              f"exorheic%={metrics['ext_percentage']:.1f}%  "
              f"({metrics['frames']} frames, {metrics['total_span_ma']:.1f} Ma)")
        ok += 1

    if not all_rows:
        print('\nNo files were processed successfully.')
        return

    # Horizontal cross-model summary (Statistic rows × model columns).
    summary_wide = pd.DataFrame(all_rows)
    summary_wide.index.name = 'Statistic'
    wide_path = os.path.join(OUT_DIR, 'all_models_D8_summary.csv')
    summary_wide.to_csv(wide_path, encoding='utf-8-sig')

    # A/C cross-model summary.
    ac_df = pd.DataFrame(ac_rows)
    ac_path = os.path.join(OUT_DIR, 'all_models_D8_A_C.csv')
    ac_df.to_csv(ac_path, index=False, encoding='utf-8-sig')

    print(f'\n{"=" * 60}')
    print(f'Batch complete: {ok}/{len(files)} files succeeded.')
    print(f'Per-model results : {OUT_DIR}')
    print(f'Cross-model table : {wide_path}')
    print(f'A/C summary table : {ac_path}')


if __name__ == '__main__':
    main()
