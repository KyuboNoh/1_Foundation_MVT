# src/gfm4mpm/sampling/likely_negatives.py
from typing import Optional
import os
import hashlib

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

@torch.no_grad()
def compute_embeddings(encoder, X, batch_size=1024, device='cuda', show_progress=False):
    encoder.eval().to(device)
    Z = []
    iterator = range(0, len(X), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Embeddings")
    for i in iterator:
        x = torch.from_numpy(X[i:i+batch_size]).to(device)
        z = encoder.encode(x).cpu()
        Z.append(z)
    return torch.cat(Z).numpy()

def pu_select_negatives_optimized_for_rotation(
    Z_all,
    all_pos_idx,
    unk_idx,
    active_pos_idx,
    filter_top_pct=0.5,
    negatives_per_pos=5,
    rng=None,
    tag: Optional[str] = None,
    *,
    return_info: bool = False,
    precomputed_distances: Optional[np.ndarray] = None,
    force_fallback: bool = False,
):
    """
    Optimized negative selection for rotation drop scenarios.
    Pre-computes distances to ALL positives once, then uses subsets for each iteration.
    
    Args:
        Z_all: All embeddings
        all_pos_idx: ALL positive indices (for distance computation)
        unk_idx: Unknown indices
        active_pos_idx: Currently active positive indices (subset of all_pos_idx)
        filter_top_pct: Percentage of top similar samples to filter
        negatives_per_pos: Number of negatives per positive
        rng: Random number generator
        tag: Cache tag
        return_info: Whether to return additional info
        precomputed_distances: Pre-computed distance matrix [n_unknown, n_all_positives]
        force_fallback: Force fallback to original method for memory safety
        
    Returns:
        Selected negative indices (and optionally info dict)
    """
    # ✅ MEMORY SAFETY: Check dataset size and fallback if too large
    n_unknowns = len(unk_idx) 
    n_all_positives = len(all_pos_idx)
    
    # Estimate memory requirements
    estimated_memory_gb = (n_unknowns * n_all_positives * 4) / (1024**3)
    
    # Fallback to original method for very large datasets or if forced
    if force_fallback or estimated_memory_gb > 10.0 or precomputed_distances is None:
        if estimated_memory_gb > 10.0:
            print(f"[pu_select_negatives_optimized] FALLBACK: Dataset too large ({estimated_memory_gb:.2f} GB), using original method")
        
        # Use original pu_select_negatives method
        return pu_select_negatives(
            Z_all=Z_all,
            pos_idx=active_pos_idx,
            unk_idx=unk_idx,
            filter_top_pct=filter_top_pct,
            negatives_per_pos=negatives_per_pos,
            rng=rng,
            tag=tag,
            return_info=return_info
        )
    if rng is None:
        rng = np.random.default_rng(1337)
    
    # If no precomputed distances, compute full distance matrix once
    if precomputed_distances is None:
        Zp_all = np.asarray(Z_all[all_pos_idx], dtype=np.float32)
        Zu = Z_all[unk_idx]
        
        # Normalize for fair distance
        Zp_norm = np.linalg.norm(Zp_all, axis=1, keepdims=True)
        Zp_norm = np.where(Zp_norm == 0.0, 1.0, Zp_norm)
        Zp_all = Zp_all / Zp_norm
        Zu_norm = np.linalg.norm(Zu, axis=1, keepdims=True)
        Zu_norm = np.where(Zu_norm == 0.0, 1.0, Zu_norm)
        Zu = Zu / Zu_norm
        
        # Compute full distance matrix [n_unknown, n_all_positives]
        full_distances = []
        step = min(1024, len(Zu))
        for i in range(0, len(Zu), step):
            chunk = Zu[i:i+step]
            dm = ((chunk[:,None,:] - Zp_all[None,:,:])**2).sum(-1)**0.5
            full_distances.append(dm)
        precomputed_distances = np.concatenate(full_distances, axis=0)
    
    # Find which columns correspond to active positives
    active_pos_mask = np.isin(all_pos_idx, active_pos_idx)
    active_distances = precomputed_distances[:, active_pos_mask]
    
    # Compute min distance to any ACTIVE positive for each unknown
    dmin = active_distances.min(axis=1)
    
    # Filter top-similar (smallest distance)
    k = int(len(dmin) * filter_top_pct)
    keep_mask = np.ones_like(dmin, dtype=bool)
    cutoff = None
    if k > 0:
        cutoff = float(np.partition(dmin, k)[:k].max())
        keep_mask &= dmin > cutoff
    kept_unk = np.array(unk_idx)[keep_mask]
    
    # Sample negatives
    effective_pos = max(len(active_pos_idx), 1)
    n_neg = min(effective_pos * negatives_per_pos, len(kept_unk))
    neg_idx = rng.choice(kept_unk, size=n_neg, replace=False)
    
    if return_info:
        info = {
            "distances": dmin,
            "kept_unknown_indices": kept_unk,
            "cutoff": cutoff,
            "keep_mask": keep_mask,
            "effective_positive_count": effective_pos,
            "precomputed_distances": precomputed_distances,  # Return for reuse
        }
        return neg_idx, info
    return neg_idx


def pu_select_negatives(
    Z_all,
    pos_idx,
    unk_idx,
    filter_top_pct=0.5,
    negatives_per_pos=5,
    rng=None,
    tag: Optional[str] = None,
    *,
    return_info: bool = False,
):
    """Select negatives from unknowns, filtering the top-similar percent to positives.

    When ``return_info`` is ``True`` an auxiliary dictionary containing the
    distance-to-positive scores and masks used during filtering is returned
    alongside the sampled negative indices.
    """
    # Create cache directory
    cache_dir = os.path.join(os.getcwd(), ".temp", "pu_distances")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache filename based on tag or data hash
    if tag is not None:
        cache_filename = f"{tag}.npz"
    else:
        # Generate hash from data parameters for unique cache key
        data_hash = hashlib.md5(f"{len(pos_idx)}_{len(unk_idx)}_{filter_top_pct}_{negatives_per_pos}".encode()).hexdigest()[:8]
        cache_filename = f"auto_{data_hash}.npz"
    
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Check if cached file exists
    file_not_found = not os.path.exists(cache_path)
    
    if file_not_found:
        if rng is None:
            rng = np.random.default_rng(1337)
        Zp = np.asarray(Z_all[pos_idx], dtype=np.float32)
        Zu = Z_all[unk_idx]

        # normalize for fair distance
        Zp_norm = np.linalg.norm(Zp, axis=1, keepdims=True)
        Zp_norm = np.where(Zp_norm == 0.0, 1.0, Zp_norm)
        Zp = Zp / Zp_norm
        Zu_norm = np.linalg.norm(Zu, axis=1, keepdims=True)
        Zu_norm = np.where(Zu_norm == 0.0, 1.0, Zu_norm)
        Zu = Zu / Zu_norm

        # Compute min distance to any positive for each unknown
        dists = []
        step = min(1024, len(Zu))
        for i in range(0, len(Zu), step):
            chunk = Zu[i:i+step]
            dm = ((chunk[:,None,:] - Zp[None,:,:])**2).sum(-1)**0.5
            dmin = dm.min(axis=1)
            dists.append(dmin)
        dmin = np.concatenate(dists)

        # filter top‑similar (smallest distance)
        k = int(len(dmin) * filter_top_pct)
        keep_mask = np.ones_like(dmin, dtype=bool)
        cutoff = None
        if k > 0:
            cutoff = float(np.partition(dmin, k)[:k].max())
            keep_mask &= dmin > cutoff
        kept_unk = np.array(unk_idx)[keep_mask]

        # sample negatives
        effective_pos = max(len(pos_idx), 1)
        n_neg = min(effective_pos * negatives_per_pos, len(kept_unk))
        neg_idx = rng.choice(kept_unk, size=n_neg, replace=False)
    
        # Save to cache file
        try:
            np.savez_compressed(
                cache_path,
                neg_idx=neg_idx,
                dmin=dmin,
                kept_unk=kept_unk,
                cutoff=cutoff,
                keep_mask=keep_mask,
                effective_pos=effective_pos,
                # Save parameters for validation
                pos_idx_len=len(pos_idx),
                unk_idx_len=len(unk_idx),
                filter_top_pct=filter_top_pct,
                negatives_per_pos=negatives_per_pos
            )
            print(f"[pu_select_negatives] Cached results to {cache_path}")
        except Exception as e:
            print(f"[pu_select_negatives] Warning: Failed to cache results: {e}")

    else:
        # Load from cache file
        try:
            cached_data = np.load(cache_path)
            
            # Validate cached data matches current parameters
            if (cached_data['pos_idx_len'] == len(pos_idx) and
                cached_data['unk_idx_len'] == len(unk_idx) and
                np.isclose(cached_data['filter_top_pct'], filter_top_pct) and
                cached_data['negatives_per_pos'] == negatives_per_pos):
                
                # Load cached results
                neg_idx = cached_data['neg_idx']
                dmin = cached_data['dmin']
                kept_unk = cached_data['kept_unk']
                cutoff = float(cached_data['cutoff']) if cached_data['cutoff'].size > 0 else None
                keep_mask = cached_data['keep_mask']
                effective_pos = int(cached_data['effective_pos'])
                
                print(f"[pu_select_negatives] Loaded cached results from {cache_path}")
            else:
                print(f"[pu_select_negatives] Cache parameters mismatch, recomputing...")
                # Remove invalid cache and recompute
                os.remove(cache_path)
                return pu_select_negatives(Z_all, pos_idx, unk_idx, filter_top_pct, 
                                         negatives_per_pos, rng, tag, return_info=return_info)
        
        except Exception as e:
            print(f"[pu_select_negatives] Warning: Failed to load cache ({e}), recomputing...")
            # Remove corrupted cache and recompute
            if os.path.exists(cache_path):
                os.remove(cache_path)
            return pu_select_negatives(Z_all, pos_idx, unk_idx, filter_top_pct, 
                                     negatives_per_pos, rng, tag, return_info=return_info)
    info = None
    if return_info:
        info = {
            "distances": dmin,
            "kept_unknown_indices": kept_unk,
            "cutoff": cutoff,
            "keep_mask": keep_mask,
            "effective_positive_count": effective_pos,
        }
        return neg_idx, info
    return neg_idx

def pu_select_negatives_augpos(
    Z_all,
    pos_idx,
    unk_idx,
    filter_top_pct=0.5,
    negatives_per_pos=5,
    rng=None,
    *,
    return_info: bool = False,
    extra_positive_embeddings: Optional[np.ndarray] = None,
):
    """Select negatives from unknowns, filtering the top-similar percent to positives + augmented positives.

    When ``return_info`` is ``True`` an auxiliary dictionary containing the
    distance-to-positive scores and masks used during filtering is returned
    alongside the sampled negative indices.
    """

    if rng is None:
        rng = np.random.default_rng(1337)
    Zp = np.asarray(Z_all[pos_idx], dtype=np.float32)
    extra_count = 0
    if extra_positive_embeddings is not None:
        extra = np.asarray(extra_positive_embeddings, dtype=np.float32)
        if extra.ndim == 1:
            extra = extra.reshape(1, -1)
        if extra.size:
            Zp = np.vstack([Zp, extra])
            extra_count = extra.shape[0]
    Zu = Z_all[unk_idx]

    # normalize for fair distance
    Zp_norm = np.linalg.norm(Zp, axis=1, keepdims=True)
    Zp_norm = np.where(Zp_norm == 0.0, 1.0, Zp_norm)
    Zp = Zp / Zp_norm
    Zu_norm = np.linalg.norm(Zu, axis=1, keepdims=True)
    Zu_norm = np.where(Zu_norm == 0.0, 1.0, Zu_norm)
    Zu = Zu / Zu_norm

    # Compute min distance to any positive for each unknown
    dists = []
    step = 4096
    for i in range(0, len(Zu), step):
        chunk = Zu[i:i+step]
        dm = ((chunk[:,None,:] - Zp[None,:,:])**2).sum(-1)**0.5
        dmin = dm.min(axis=1)
        dists.append(dmin)
    dmin = np.concatenate(dists)

    # filter top‑similar (smallest distance)
    k = int(len(dmin) * filter_top_pct)
    keep_mask = np.ones_like(dmin, dtype=bool)
    cutoff = None
    if k > 0:
        cutoff = float(np.partition(dmin, k)[:k].max())
        keep_mask &= dmin > cutoff
    kept_unk = np.array(unk_idx)[keep_mask]

    # sample negatives
    effective_pos = max(len(pos_idx) + extra_count, 1)
    n_neg = min(effective_pos * negatives_per_pos, len(kept_unk))
    neg_idx = rng.choice(kept_unk, size=n_neg, replace=False)
    if return_info:
        info = {
            "distances": dmin,
            "kept_unknown_indices": kept_unk,
            "cutoff": cutoff,
            "keep_mask": keep_mask,
            "effective_positive_count": effective_pos,
        }
        return neg_idx, info
    return neg_idx

def precompute_distances_per_positive(
    Z_all,
    all_pos_idx,
    unk_idx,
    tag: Optional[str] = None,
    use_float16: bool = True,
    chunk_size: int = 4096,  # ✅ NEW: Process unknowns in chunks
) -> str:
    """
    Pre-compute distances separately for each positive sample (OPTIMIZED).
    Saves individual distance files to avoid memory issues.
    
    Args:
        Z_all: All embeddings
        all_pos_idx: ALL positive indices
        unk_idx: Unknown indices
        tag: Cache tag for saving/loading
        use_float16: Use float16 for memory savings
        chunk_size: Process unknowns in chunks to manage memory
        
    Returns:
        Directory path containing individual distance files
    """
    import time
    import os
    import hashlib
    
    # ✅ STEP 1: Ensure data is on GPU as tensor
    print(f"[Compute_Individual_Distances] ===== PERFORMANCE CHECK =====")
    start_total = time.time()
    
    # Check input type and device
    if isinstance(Z_all, torch.Tensor):
        device = Z_all.device
        # print(f"[Compute_Individual_Distances] Z_all is already a tensor on {device}")
        
        # ✅ Ensure it's on GPU if available
        if device.type == 'cpu' and torch.cuda.is_available():
            # print(f"[Compute_Individual_Distances] Moving Z_all from CPU to GPU...")
            Z_all = Z_all.cuda()
            device = Z_all.device
    
    elif isinstance(Z_all, np.ndarray):
        # print(f"[Compute_Individual_Distances] Z_all is NumPy array")
        # print(f"[Compute_Individual_Distances] Converting to GPU tensor...")
        t0 = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Z_all = torch.from_numpy(Z_all).float().to(device)
    else:
        raise TypeError(f"Unsupported Z_all type: {type(Z_all)}")

    
    n_samples, embedding_dim = Z_all.shape

    
    # ✅ STEP 2: Convert indices to GPU tensors
    if isinstance(unk_idx, list):
        unk_idx_tensor = torch.tensor(unk_idx, dtype=torch.long, device=device)
    elif isinstance(unk_idx, np.ndarray):
        unk_idx_tensor = torch.from_numpy(unk_idx).long().to(device)
    else:
        unk_idx_tensor = unk_idx.to(device)
    
    # ✅ STEP 3: Extract and pre-normalize ALL unknowns ONCE on GPU
    t0 = time.time()
    
    Zu = Z_all[unk_idx_tensor]  # Shape: (n_unknowns, embedding_dim)
    
    # Normalize on GPU (FAST)
    Zu_norm = torch.linalg.norm(Zu, dim=1, keepdim=True)
    Zu_norm = torch.where(Zu_norm == 0.0, torch.ones_like(Zu_norm), Zu_norm)
    Zu = Zu / Zu_norm
    
    if use_float16:
        Zu = Zu.half()
        
    # ✅ STEP 4: Setup cache directory
    cache_dir = os.path.join(os.getcwd(), ".temp", "individual_distances")
    if tag is not None:
        cache_subdir = os.path.join(cache_dir, f"{tag}_individual")
    else:
        data_hash = hashlib.md5(f"{len(all_pos_idx)}_{len(unk_idx)}_individual".encode()).hexdigest()[:8]
        cache_subdir = os.path.join(cache_dir, f"auto_individual_{data_hash}")
    
    os.makedirs(cache_subdir, exist_ok=True)
    
    # ✅ STEP 5: Check cache
    all_files_exist = all(
        os.path.exists(os.path.join(cache_subdir, f"pos_{pos_idx}_distances.npy"))
        for pos_idx in all_pos_idx
    )
    
    if all_files_exist:
        print(f"[Compute_Individual_Distances] ✅ Found all cached files in {cache_subdir}")
        return cache_subdir
    
    # ✅ STEP 6: Compute distances on GPU (OPTIMIZED)
    print(f"[Compute_Individual_Distances] Computing distances on GPU...")
    
    computed_count = 0
    t_compute_start = time.time()
    n_unknowns = len(Zu)
    
    for i, pos_idx in enumerate(all_pos_idx):
        distance_file = os.path.join(cache_subdir, f"pos_{pos_idx}_distances.npy")
        
        if os.path.exists(distance_file):
            continue
        
        t_pos_start = time.time()
        
        # ✅ OPTIMIZED: Get and normalize positive on GPU
        zp = Z_all[pos_idx].unsqueeze(0)  # Shape: (1, embedding_dim)
        zp_norm = torch.linalg.norm(zp, dim=1, keepdim=True)
        zp_norm = torch.where(zp_norm == 0.0, torch.ones_like(zp_norm), zp_norm)
        zp = zp / zp_norm
        
        if use_float16:
            zp = zp.half()
        
        # ✅ CRITICAL OPTIMIZATION: Compute distances in large chunks on GPU
        all_distances = []
        
        for chunk_start in range(0, n_unknowns, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_unknowns)
            zu_chunk = Zu[chunk_start:chunk_end]  # Already on GPU!
            
            # ✅ FASTEST GPU DISTANCE: Euclidean norm
            # Use broadcasting: (chunk_size, dim) - (1, dim) = (chunk_size, dim)
            distances_chunk = torch.norm(zu_chunk - zp, dim=1)  # Shape: (chunk_size,)
            
            all_distances.append(distances_chunk)
        
        # Combine chunks and move to CPU for saving
        distances = torch.cat(all_distances).cpu().numpy()
        
        if use_float16:
            distances = distances.astype(np.float16)
        
        # Save
        np.save(distance_file, distances)
        computed_count += 1
        
        t_pos_elapsed = time.time() - t_pos_start
        
        # ✅ Progress logging with timing
        if (i + 1) % 100 == 0 or (i + 1) == len(all_pos_idx):
            elapsed = time.time() - t_compute_start
            avg_time_per_pos = elapsed / (i + 1)
            remaining_pos = len(all_pos_idx) - (i + 1)
            eta_seconds = remaining_pos * avg_time_per_pos
            
            progress_pct = (i + 1) / len(all_pos_idx) * 100
            
            print(f"[Compute_Individual_Distances] Progress: {progress_pct:.1f}% ({i+1}/{len(all_pos_idx)}) "
                  f"| Avg: {avg_time_per_pos:.3f}s/pos | Last: {t_pos_elapsed:.3f}s | ETA: {eta_seconds/60:.1f}min")
    
    total_time = time.time() - start_total
    compute_time = time.time() - t_compute_start
    
    print(f"[Compute_Individual_Distances] ✅ Completed!")
    
    # ✅ Cleanup GPU memory
    if device.type == 'cuda':
        del Zu, zp, Z_all
        torch.cuda.empty_cache()
    
    return cache_subdir

def pu_select_negatives_from_individual_distances(
    individual_distances_dir: str,
    all_pos_idx,
    unk_idx,
    active_pos_idx,
    filter_top_pct=0.5,
    negatives_per_pos=5,
    rng=None,
    return_info: bool = False,
):
    """
    Select negatives using pre-computed individual distance files.
    
    Args:
        individual_distances_dir: Directory containing individual distance files
        all_pos_idx: ALL positive indices
        unk_idx: Unknown indices  
        active_pos_idx: Currently active positive indices
        filter_top_pct: Percentage of top similar samples to filter
        negatives_per_pos: Number of negatives per positive
        rng: Random number generator
        return_info: Whether to return additional info
        
    Returns:
        Selected negative indices
    """
    if rng is None:
        rng = np.random.default_rng(1337)
    
    # Load distances for active positives only
    active_distances = []
    for pos_idx in active_pos_idx:
        distance_file = os.path.join(individual_distances_dir, f"pos_{pos_idx}_distances.npy")
        if os.path.exists(distance_file):
            distances = np.load(distance_file)
            active_distances.append(distances)
        else:
            print(f"[pu_select_negatives_individual] Warning: Missing distance file for positive {pos_idx}")
    
    if not active_distances:
        raise ValueError("No distance files found for active positives")
    
    # Stack distances and compute minimum distance to any active positive
    distance_matrix = np.stack(active_distances, axis=1)  # [n_unknown, n_active_positives]
    dmin = distance_matrix.min(axis=1)
    
    # Filter top-similar (smallest distance)
    k = int(len(dmin) * filter_top_pct)
    keep_mask = np.ones_like(dmin, dtype=bool)
    cutoff = None
    if k > 0:
        cutoff = float(np.partition(dmin, k)[:k].max())
        keep_mask &= dmin > cutoff
    kept_unk = np.array(unk_idx)[keep_mask]
    
    # Sample negatives
    effective_pos = max(len(active_pos_idx), 1)
    n_neg = min(effective_pos * negatives_per_pos, len(kept_unk))
    neg_idx = rng.choice(kept_unk, size=n_neg, replace=False)
    
    # print(f"[pu_select_negatives_individual] Selected {len(neg_idx)} negatives from {len(kept_unk)} filtered unknowns")
    
    if return_info:
        info = {
            "distances": dmin,
            "kept_unknown_indices": kept_unk,
            "cutoff": cutoff,
            "keep_mask": keep_mask,
            "effective_positive_count": effective_pos,
        }
        return neg_idx, info
    return neg_idx

def precompute_distances_for_rotation_drops(
    Z_all,
    all_pos_idx,
    unk_idx,
    tag: Optional[str] = None,
) -> np.ndarray:
    """
    Pre-compute distance matrix for rotation drop optimization.
    Computes distances from all unknowns to ALL positives once.
    
    Args:
        Z_all: All embeddings
        all_pos_idx: ALL positive indices
        unk_idx: Unknown indices
        tag: Cache tag for saving/loading
        
    Returns:
        Distance matrix [n_unknown, n_all_positives]
    """
    # Create cache directory
    cache_dir = os.path.join(os.getcwd(), ".temp", "embedding_distances")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache filename
    if tag is not None:
        cache_filename = f"{tag}_full_distances.npz"
    else:
        data_hash = hashlib.md5(f"{len(all_pos_idx)}_{len(unk_idx)}_full".encode()).hexdigest()[:8]
        cache_filename = f"auto_full_{data_hash}.npz"
    
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Check if cached file exists
    if os.path.exists(cache_path):
        try:
            cached_data = np.load(cache_path)
            
            # Validate cached data
            if (cached_data['pos_idx_len'] == len(all_pos_idx) and
                cached_data['unk_idx_len'] == len(unk_idx)):
                
                print(f"[Compute_Embedding_Distances] Loaded cached distances from {cache_path}")
                return cached_data['distance_matrix']
            else:
                print(f"[Compute_Embedding_Distances] Cache size mismatch, recomputing...")
                os.remove(cache_path)
        except Exception as e:
            print(f"[Compute_Embedding_Distances] Failed to load cache ({e}), recomputing...")
            if os.path.exists(cache_path):
                os.remove(cache_path)
    
    # Compute full distance matrix
    print(f"[Compute_Embedding_Distances] Computing distances: {len(unk_idx)} unknowns × {len(all_pos_idx)} positives")
    
    Zp_all = np.asarray(Z_all[all_pos_idx], dtype=np.float32)
    Zu = Z_all[unk_idx]
    
    # Normalize for fair distance
    Zp_norm = np.linalg.norm(Zp_all, axis=1, keepdims=True)
    Zp_norm = np.where(Zp_norm == 0.0, 1.0, Zp_norm)
    Zp_all = Zp_all / Zp_norm
    Zu_norm = np.linalg.norm(Zu, axis=1, keepdims=True)
    Zu_norm = np.where(Zu_norm == 0.0, 1.0, Zu_norm)
    Zu = Zu / Zu_norm
    
    # ✅ MEMORY-EFFICIENT: Compute distance matrix in smaller chunks
    # Estimate memory usage and adjust chunk size accordingly
    n_unknowns = len(Zu)
    n_positives = len(Zp_all)
    embedding_dim = Zu.shape[1] if len(Zu) > 0 else 0
    
    # Estimate memory for full distance matrix (in GB)
    estimated_memory_gb = (n_unknowns * n_positives * 4) / (1024**3)  # 4 bytes per float32

    step = min(1024, len(Zu))
    
    # Compute full distance matrix in chunks to manage memory
    full_distances = []
    for i in range(0, len(Zu), step):
        chunk = Zu[i:i+step]
        # Compute distances from this chunk to all positives
        dm = ((chunk[:,None,:] - Zp_all[None,:,:])**2).sum(-1)**0.5
        full_distances.append(dm)
        
        # Progress logging for large computations
        if len(Zu) > 10000 and (i // step) % 100 == 0:
            progress = (i + step) / len(Zu) * 100
            print(f"[Compute_Embedding_Distances] Progress: {progress:.1f}% ({i + step}/{len(Zu)})")
    
    distance_matrix = np.concatenate(full_distances, axis=0)
    
    # Save to cache
    try:
        np.savez_compressed(
            cache_path,
            distance_matrix=distance_matrix,
            pos_idx_len=len(all_pos_idx),
            unk_idx_len=len(unk_idx),
        )
        print(f"[Compute_Embedding_Distances] Cached distance matrix to {cache_path}")
    except Exception as e:
        print(f"[Compute_Embedding_Distances] Warning: Failed to cache distances: {e}")
    
    return distance_matrix