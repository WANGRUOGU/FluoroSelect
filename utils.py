import yaml
import numpy as np
import pulp


# --- CBC solver: no time/gap limits (guarantee optimality), quiet log ---
def _make_cbc_exact():
    try:
        return pulp.PULP_CBC_CMD(msg=False, mip=True)
    except TypeError:
        return pulp.PULP_CBC_CMD(msg=False)


_SOLVER = _make_cbc_exact()


# ====================== I/O ======================
def load_dyes_yaml(path):
    """
    Load dyes.yaml -> (wavelengths, dye_db).

    dye_db[name] = {
        "emission": np.array(W,),
        "excitation": np.array(W,),
        "quantum_yield": float|None,
        "extinction_coeff": float|None,
    }

    Missing QY will be filled with the mean of available QYs.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    wl = np.array(data["wavelengths"], dtype=float)
    dye_db = {}

    for name, rec in data["dyes"].items():
        em = np.array(rec.get("emission", []), dtype=float)
        ex = np.array(rec.get("excitation", []), dtype=float)
        qy = rec.get("quantum_yield", None)
        ec = rec.get("extinction_coeff", None)
        dye_db[name] = dict(
            emission=em,
            excitation=ex,
            quantum_yield=qy,
            extinction_coeff=ec,
        )

    qys = [v["quantum_yield"] for v in dye_db.values() if v.get("quantum_yield") is not None]
    mean_qy = float(np.mean(qys)) if len(qys) else 1.0

    for v in dye_db.values():
        if v.get("quantum_yield") is None:
            v["quantum_yield"] = mean_qy

    return wl, dye_db


def load_probe_fluor_map(path):
    """
    Accepts:
    - top-level list of {name: , fluors: [...]}
    - or {probes: [...] } with same structure.

    Returns dict[probe] -> list[fluor].
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("probes"), list):
        items = data["probes"]
    else:
        return {}

    mapping = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name", "")).strip()
        fls = it.get("fluors", []) or []
        if not name:
            continue
        if not isinstance(fls, (list, tuple)):
            fls = [str(fls).strip()] if str(fls).strip() else []
        mapping[name] = [str(x).strip() for x in fls if str(x).strip()]

    return mapping


# =================== Linear-algebra helpers ===================
def _safe_l2norm_cols(E):
    """L2-normalize each column with a numerical guard."""
    denom = np.linalg.norm(E, axis=0, keepdims=True) + 1e-12
    return E / denom


def cosine_similarity_matrix(E):
    """
    Cosine similarity among columns of E.
    Diagonal is set to 0 so it is easy to take pairwise maxima.
    """
    norms = np.linalg.norm(E, axis=0) + 1e-12
    G = (E.T @ E) / np.outer(norms, norms)
    np.fill_diagonal(G, 0.0)
    return G

def spectral_overlap_similarity_matrix(E):
    """
    Area-normalized spectral overlap among columns of E.

    Each spectrum is first clipped to nonnegative values and normalized
    to unit area. Similarity is sum(min(p_i, p_j)).
    Range: 0 to 1. Larger means more overlap.
    """
    X = np.maximum(np.asarray(E, dtype=float), 0.0)
    denom = np.sum(X, axis=0, keepdims=True) + 1e-12
    P = X / denom

    G = np.minimum(P[:, :, None], P[:, None, :]).sum(axis=0)
    G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
    G = np.clip(G, 0.0, 1.0)
    np.fill_diagonal(G, 0.0)

    return G


def pearson_similarity_matrix(E):
    """
    Pearson correlation similarity among columns of E.

    The raw Pearson correlation is mapped from [-1, 1] to [0, 1]
    so that larger values always mean more similar.
    """
    X = np.asarray(E, dtype=float)
    X = X - np.mean(X, axis=0, keepdims=True)

    norms = np.linalg.norm(X, axis=0) + 1e-12
    G = (X.T @ X) / np.outer(norms, norms)

    G = (G + 1.0) / 2.0
    G = np.nan_to_num(G, nan=0.0, posinf=1.0, neginf=0.0)
    G = np.clip(G, 0.0, 1.0)
    np.fill_diagonal(G, 0.0)

    return G


def spectral_angle_similarity_matrix(E):
    """
    Spectral-angle similarity among columns of E.

    This is based on the spectral angle mapper:
        angle = arccos(cosine)

    We convert angle to a similarity score:
        similarity = 1 - angle / (pi/2)

    For nonnegative spectra, this gives a score close to [0, 1].
    Larger means more similar.
    """
    C = cosine_similarity_matrix(E)
    C = np.clip(C, -1.0, 1.0)

    angle = np.arccos(C)
    G = 1.0 - angle / (np.pi / 2.0)

    G = np.nan_to_num(G, nan=0.0, posinf=1.0, neginf=0.0)
    G = np.clip(G, 0.0, 1.0)
    np.fill_diagonal(G, 0.0)

    return G


def similarity_matrix(E, metric="Cosine similarity"):
    """
    Compute a pairwise similarity/confusability matrix.

    All supported metrics are scaled so that larger values mean
    more similar and therefore worse for fluorophore separation.
    """
    metric = metric or "Cosine similarity"

    if metric == "Cosine similarity":
        return cosine_similarity_matrix(E)

    if metric == "Spectral overlap":
        return spectral_overlap_similarity_matrix(E)

    if metric == "Pearson correlation":
        return pearson_similarity_matrix(E)

    if metric == "Spectral angle similarity":
        return spectral_angle_similarity_matrix(E)

    raise ValueError(f"Unknown similarity metric: {metric}")

def top_k_pairwise(S, labels_pair, k=10):
    """
    Given NxN cosine matrix S, return top-k pairs.
    Each item: (value, label_i, label_j), sorted descending by value.
    """
    N = S.shape[0]
    iu = np.triu_indices(N, k=1)
    vals = S[iu]
    if vals.size == 0:
        return []

    order = np.argsort(-vals)[: min(k, vals.size)]
    out = []
    for idx in order:
        i = iu[0][idx]
        j = iu[1][idx]
        out.append((float(vals[idx]), labels_pair[i], labels_pair[j]))
    return out


# =================== Spectra builders ===================
def build_emission_only_matrix(wl, dye_db, groups):
    """
    Build peak-normalized emission-only matrix for optimization.

    Returns:
    - E_norm: W x N, L2-normalized by column
    - labels_pair: list[str] "Probe – Fluor"
    - idx_groups: list[list[int]], column indices per probe group
    """
    W = len(wl)
    cols, labels, idx_groups = [], [], []
    col_id = 0

    for probe, cand_list in groups.items():
        idxs = []
        for fluor in cand_list:
            rec = dye_db.get(fluor)
            if rec is None:
                continue
            em = np.array(rec["emission"], dtype=float)
            if em.size != W:
                continue
            m = np.max(em) if np.max(em) > 0 else 1.0
            em_peak = em / m
            cols.append(em_peak)
            labels.append(f"{probe} – {fluor}")
            idxs.append(col_id)
            col_id += 1
        if idxs:
            idx_groups.append(idxs)

    if not cols:
        return np.zeros((W, 0)), [], []

    E = np.stack(cols, axis=1)
    E_norm = _safe_l2norm_cols(E)
    return E_norm, labels, idx_groups


def _nearest_idx_from_grid(wl, lam):
    """Assuming 1 nm grid starting at wl[0]; pick nearest index."""
    idx = int(round(lam - wl[0]))
    if idx < 0:
        idx = 0
    if idx >= len(wl):
        idx = len(wl) - 1
    return idx


def _segments_from_lasers(wl, lasers_sorted):
    """Segments [lo, hi) defined by sorted laser wavelengths."""
    segs = []
    for i, l in enumerate(lasers_sorted):
        lo = l
        hi = lasers_sorted[i + 1] if i + 1 < len(lasers_sorted) else wl[-1] + 1
        segs.append((lo, hi))
    return segs


def _interp_at(w, y, x):
    """1D linear interpolation on a discrete grid, clamped."""
    if x <= w[0]:
        return float(y[0])
    if x >= w[-1]:
        return float(y[-1])
    i = int(np.searchsorted(w, x)) - 1
    t = (x - w[i]) / (w[i + 1] - w[i])
    return float(y[i] * (1 - t) + y[i + 1] * t)


def derive_powers_simultaneous(wl, dye_db, selection_labels, laser_wavelengths):
    """
    Calibrate laser powers in Simultaneous mode.
    """
    lam = np.array(sorted(set(float(l) for l in laser_wavelengths)), dtype=float)
    W = len(wl)
    if lam.size == 0:
        return [0.0] * 0, 0.0

    fluor_names = [s.split(" – ", 1)[1] for s in selection_labels]
    recs = []
    for f in fluor_names:
        rec = dye_db.get(f)
        if rec is None:
            continue
        em = rec.get("emission")
        ex = rec.get("excitation")
        if em is None or ex is None or len(em) != W or len(ex) != W:
            continue
        recs.append(rec)

    if not recs:
        return [0.0] * len(lam), 0.0

    segs = _segments_from_lasers(wl, lam)

    def seg_peak(rec, seg_index):
        lo, hi = segs[seg_index]
        loi = _nearest_idx_from_grid(wl, lo)
        hii = _nearest_idx_from_grid(wl, hi - 1) + 1
        if loi >= hii:
            return 0.0
        return float(np.max(rec["emission"][loi:hii]))

    def coef_at(rec, l):
        ex = rec["excitation"]
        qy = rec.get("quantum_yield", None)
        ec = rec.get("extinction_coeff", None)
        if ex is None or len(ex) != W or qy is None:
            return 0.0
        ex_l = _interp_at(wl, ex, l)
        return float(ex_l * qy * (ec if ec is not None else 1.0))

    seg_has_peak = [False] * len(segs)
    for rec in recs:
        em = rec["emission"]
        if em is None or len(em) != W:
            continue
        jmax = int(np.argmax(em))
        lam_peak = wl[jmax]
        for s, (lo, hi) in enumerate(segs):
            if lo <= lam_peak < hi:
                seg_has_peak[s] = True
                break

    peak_segs = [s for s, u in enumerate(seg_has_peak) if u]
    P = np.zeros(len(lam), dtype=float)
    if not peak_segs:
        return P.tolist(), 0.0

    s0 = peak_segs[0]
    P[s0] = 1.0

    B = 0.0
    for rec in recs:
        m0 = seg_peak(rec, s0)
        if m0 <= 0:
            continue
        k0 = coef_at(rec, lam[s0]) * P[s0]
        val = m0 * k0
        if val > B:
            B = val

    if B <= 0.0:
        return P.tolist(), 0.0

    for s in peak_segs[1:]:
        cand_c = []
        for rec in recs:
            m_seg = seg_peak(rec, s)
            if m_seg <= 0.0:
                continue
            pre_j = 0.0
            for m_idx in range(s):
                if P[m_idx] == 0.0:
                    continue
                k_prev = coef_at(rec, lam[m_idx])
                pre_j += k_prev * P[m_idx]
            k_js = coef_at(rec, lam[s])
            if k_js <= 0.0:
                continue
            c_j = (B / m_seg - pre_j) / k_js
            if c_j > 0.0:
                cand_c.append(c_j)
        P[s] = float(max(0.0, min(cand_c))) if cand_c else 0.0

    return P.tolist(), float(B)


def derive_powers_separate(wl, dye_db, selection_labels, laser_wavelengths):
    """
    Separate-mode calibration.
    """
    lam = np.array(sorted(laser_wavelengths), dtype=float)
    W = len(wl)
    fls = [s.split(" – ", 1)[1] for s in selection_labels]
    recs = [dye_db[f] for f in fls if f in dye_db]

    def coef(rec, l):
        ex = rec["excitation"]
        qy = rec["quantum_yield"]
        ec = rec["extinction_coeff"]
        if ex is None or len(ex) != W or qy is None:
            return 0.0
        ex_l = _interp_at(wl, ex, l)
        return float(ex_l * qy * (ec if ec is not None else 1.0))

    M = []
    for l in lam:
        peaks = []
        for r in recs:
            em = r["emission"]
            if em is None or len(em) != W:
                continue
            peaks.append(np.max(em) * coef(r, l))
        M.append(max(peaks) if peaks else 0.0)

    M = np.array(M, dtype=float)
    P = np.zeros_like(M)
    if M.size == 0:
        return [1.0] * 0, 1.0

    P[0] = 1.0
    B = float(M[0])
    for i in range(1, len(M)):
        P[i] = float(B / M[i]) if M[i] > 0 else 0.0

    return [float(x) for x in P], B


def build_effective_with_lasers(wl, dye_db, groups, laser_wavelengths, mode, powers):
    """
    Build effective spectra for ALL candidates.

    Returns:
    - E_raw
    - E_norm
    - labels_pair
    - idx_groups
    """
    W = len(wl)
    lam = np.array(sorted(laser_wavelengths), dtype=float)
    pw = np.array(powers, dtype=float)

    def _nearest_idx_from_grid_local(wl_, lam_):
        idx = int(round(lam_ - wl_[0]))
        if idx < 0:
            idx = 0
        if idx >= len(wl_):
            idx = len(wl_) - 1
        return idx

    def _segments_from_lasers_local(wl_, lasers_sorted):
        segs_ = []
        for i_, l_ in enumerate(lasers_sorted):
            lo_ = l_
            hi_ = lasers_sorted[i_ + 1] if i_ + 1 < len(lasers_sorted) else wl_[-1] + 1
            segs_.append((lo_, hi_))
        return segs_

    cols, labels, idx_groups = [], [], []
    col_id = 0

    if mode == "Separate":
        for probe, cand_list in groups.items():
            idxs = []
            for fluor in cand_list:
                rec = dye_db.get(fluor)
                if rec is None:
                    continue
                em = rec["emission"]
                ex = rec["excitation"]
                qy = rec["quantum_yield"]
                ec = rec["extinction_coeff"]
                if em is None or ex is None or len(em) != W or len(ex) != W:
                    continue
                per_laser_blocks = []
                for i, l in enumerate(lam):
                    k = _interp_at(wl, ex, l) * qy * (ec if ec is not None else 1.0)
                    k *= pw[i]
                    per_laser_blocks.append(em * k)
                eff_concat = np.concatenate(per_laser_blocks, axis=0)
                cols.append(eff_concat)
                labels.append(f"{probe} – {fluor}")
                idxs.append(col_id)
                col_id += 1
            if idxs:
                idx_groups.append(idxs)

        if not cols:
            Z = np.zeros((W * max(1, len(lam)), 0))
            return Z, Z, [], []

        E_raw = np.stack(cols, axis=1)
        denom = np.linalg.norm(E_raw, axis=0, keepdims=True) + 1e-12
        E_norm = E_raw / denom
        return E_raw, E_norm, labels, idx_groups

    # Simultaneous mode: per-segment cumulative coefficient
    segs = _segments_from_lasers_local(wl, lam)
    for probe, cand_list in groups.items():
        idxs = []
        for fluor in cand_list:
            rec = dye_db.get(fluor)
            if rec is None:
                continue
            em = rec["emission"]
            ex = rec["excitation"]
            qy = rec["quantum_yield"]
            ec = rec["extinction_coeff"]
            if em is None or ex is None or len(em) != W or len(ex) != W:
                continue
            eff = np.zeros(W, dtype=float)
            for i, (lo, hi) in enumerate(segs):
                loi = _nearest_idx_from_grid_local(wl, lo)
                hii = _nearest_idx_from_grid_local(wl, hi - 1) + 1
                if loi >= hii:
                    continue
                total_k = 0.0
                for m in range(i + 1):
                    ex_l = _interp_at(wl, ex, lam[m])
                    total_k += ex_l * qy * (ec if ec is not None else 1.0) * pw[m]
                eff[loi:hii] += em[loi:hii] * total_k
            cols.append(eff)
            labels.append(f"{probe} – {fluor}")
            idxs.append(col_id)
            col_id += 1
        if idxs:
            idx_groups.append(idxs)

    if not cols:
        Z = np.zeros((W, 0))
        return Z, Z, [], []

    E_raw = np.stack(cols, axis=1)
    denom = np.linalg.norm(E_raw, axis=0, keepdims=True) + 1e-12
    E_norm = E_raw / denom
    return E_raw, E_norm, labels, idx_groups


# =================== Global-unique constraint ===================
def _unique_dye_constraints(prob, x_vars, labels_pair, groups, fluor_names):
    """
    Enforce each fluorophore can be used at most once globally.
    Works for by-probes mode; not applied in pool mode.
    """
    dye_to_cols = {}
    for j, d in enumerate(fluor_names):
        dye_to_cols.setdefault(d, []).append(j)
    for d, cols in dye_to_cols.items():
        prob += pulp.lpSum(x_vars[j] for j in cols) <= 1, f"Unique_{d}"


# =================== Optimization helpers ===================
def _pick_integral_from_relaxed(x_vars, idx_groups):
    """Safety: if a MIP solution leaves fractional x, pick argmax within each group."""
    xvals = np.array([(v.value() or 0.0) for v in x_vars], dtype=float)
    sel = []
    for idxs in idx_groups:
        if not idxs:
            continue
        j_local = int(np.argmax(xvals[idxs]))
        sel.append(int(idxs[j_local]))
    return sel


def _normalize_fixed_allowed(N, fixed_indices=None, allowed_indices=None):
    """
    Convert fixed/allowed indices to validated sets.

    fixed_indices: selected columns forced to 1.
    allowed_indices: columns allowed for additional selection.
    Fixed columns are allowed even if not included in allowed_indices.
    """
    fixed = {int(j) for j in (fixed_indices or []) if 0 <= int(j) < N}
    allowed = (
        {int(j) for j in allowed_indices if 0 <= int(j) < N}
        if allowed_indices is not None
        else set(range(N))
    )
    selectable = allowed | fixed
    return fixed, allowed, selectable


def _add_fixed_allowed_constraints(prob, x, N, fixed_indices=None, allowed_indices=None):
    fixed, allowed, selectable = _normalize_fixed_allowed(N, fixed_indices, allowed_indices)

    for j in fixed:
        prob += x[j] == 1, f"Fixed_{j}"

    for j in range(N):
        if j not in selectable:
            prob += x[j] == 0, f"NotAllowed_{j}"

    return fixed, allowed, selectable


# =================== Optimization: minimax layer ===================
def solve_minimax_layer(
    E_norm,
    idx_groups,
    labels_pair,
    enforce_unique=True,
    required_count: int | None = None,
    fixed_indices=None,
    allowed_indices=None,
    similarity_metric="Cosine similarity",
):
    """
    Minimize the maximum pairwise cosine among selected columns.

    Modes:
    - required_count is None: by-probes mode, pick exactly one per group.
    - required_count is int: pool mode, pick exactly required_count from the union set.

    In pool mode, fixed_indices are forced into the final selection, and
    allowed_indices restrict the additional candidates.
    """
    N = E_norm.shape[1]
    C = similarity_matrix(E_norm, metric=similarity_metric)

    fixed, _, _ = _normalize_fixed_allowed(N, fixed_indices, allowed_indices)
    if required_count is not None and len(fixed) > int(required_count):
        raise ValueError(
            f"Number of fixed fluorophores ({len(fixed)}) cannot exceed "
            f"required_count ({required_count})."
        )

    prob = pulp.LpProblem("minimax", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Binary") for j in range(N)]
    y = {}

    for i in range(N):
        for j in range(i + 1, N):
            y[(i, j)] = pulp.LpVariable(f"y_{i}_{j}", lowBound=0, upBound=1)

    t = pulp.LpVariable("t", lowBound=0)

    _add_fixed_allowed_constraints(prob, x, N, fixed_indices, allowed_indices)

    # selection constraints
    if required_count is None:
        for g, idxs in enumerate(idx_groups):
            prob += pulp.lpSum(x[j] for j in idxs) == 1, f"OnePerGroup_{g}"
        if enforce_unique:
            fluor_names = [s.split(" – ", 1)[1] for s in labels_pair]
            _unique_dye_constraints(prob, x, labels_pair, idx_groups, fluor_names)
    else:
        prob += pulp.lpSum(x) == int(required_count), "PickN"

    # y linking + t >= C_ij * y_ij
    for (i, j), yij in y.items():
        prob += yij <= x[i]
        prob += yij <= x[j]
        prob += yij >= x[i] + x[j] - 1
        prob += t >= float(C[i, j]) * yij

    prob += t
    _ = prob.solve(_SOLVER)

    if required_count is None:
        x_star = _pick_integral_from_relaxed(x, idx_groups)
    else:
        xv = np.array([(v.value() or 0.0) for v in x])
        x_star = [j for j in range(N) if xv[j] > 0.5]
        if len(x_star) != int(required_count):
            x_star = list(np.argsort(-xv)[: int(required_count)])

    t_val = float(t.value() or 0.0)
    return x_star, t_val


# =================== Optimization: lexicographic ===================
def solve_lexicographic_k(
    E_norm,
    idx_groups,
    labels_pair,
    levels: int = 10,
    enforce_unique: bool = True,
    required_count: int | None = None,
    fixed_indices=None,
    allowed_indices=None,
    similarity_metric="Cosine similarity",
):
    """
    True lexicographic minimization over pairwise cosines.

    Modes:
    - required_count=None: by-probes mode, one per group plus global unique.
    - required_count=int: pool mode, select N from the union set.

    Optional pool-mode constraints:
    - fixed_indices: columns forced into the final selected set.
    - allowed_indices: columns allowed as additional candidates.
    """
    N = E_norm.shape[1]
    if N == 0:
        return [], []

    fixed, _, _ = _normalize_fixed_allowed(N, fixed_indices, allowed_indices)
    if required_count is not None and len(fixed) > int(required_count):
        raise ValueError(
            f"Number of fixed fluorophores ({len(fixed)}) cannot exceed "
            f"required_count ({required_count})."
        )

    C = similarity_matrix(E_norm, metric=similarity_metric)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    P = len(pairs)

    if P == 0:
        if required_count is None:
            sel = [g[0] for g in idx_groups if g]
        else:
            sel = list(range(min(int(required_count), N)))
        return sel, []

    K = int(max(1, min(levels, P)))
    eps = 1e-6
    best_layers, x_star_last = [], None

    # single-group view for reading/writing x uniformly
    single_group = idx_groups if required_count is None else [list(range(N))]

    for k in range(1, K + 1):
        prob = pulp.LpProblem(f"lexi_k{k}", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Binary") for j in range(N)]
        y = {
            (i, j): pulp.LpVariable(f"y_{i}_{j}", lowBound=0, upBound=1)
            for (i, j) in pairs
        }
        z = {(i, j): pulp.LpVariable(f"z_{i}_{j}", lowBound=0) for (i, j) in pairs}

        _add_fixed_allowed_constraints(prob, x, N, fixed_indices, allowed_indices)

        # selection constraints
        if required_count is None:
            for g, idxs in enumerate(idx_groups):
                prob += pulp.lpSum(x[j] for j in idxs) == 1, f"OnePerGroup_{g}"
            if enforce_unique:
                fluor_names = [s.split(" – ", 1)[1] for s in labels_pair]
                _unique_dye_constraints(prob, x, labels_pair, idx_groups, fluor_names)
        else:
            prob += pulp.lpSum(x) == int(required_count), "PickN"

        # linking
        for (i, j) in pairs:
            yij = y[(i, j)]
            prob += yij <= x[i]
            prob += yij <= x[j]
            prob += yij >= x[i] + x[j] - 1

            cij = float(C[i, j])
            zij = z[(i, j)]
            prob += zij <= cij * yij
            prob += zij >= cij * yij

        # level-1
        t1 = pulp.LpVariable("t1", lowBound=0)
        for (i, j) in pairs:
            prob += t1 >= z[(i, j)]
        if len(best_layers) > 0:
            prob += t1 <= float(best_layers[0]) + eps

        # levels 2..k
        lam, mu = {}, {}
        if k >= 2:
            for r in range(2, k + 1):
                lam[r] = pulp.LpVariable(f"lam{r}", lowBound=0)
                mu[r] = {
                    (i, j): pulp.LpVariable(f"mu{r}_{i}_{j}", lowBound=0)
                    for (i, j) in pairs
                }
                for (i, j) in pairs:
                    prob += mu[r][(i, j)] >= z[(i, j)] - lam[r]
                if (r - 1) < len(best_layers):
                    prob += lam[r] <= float(best_layers[r - 1]) + eps

        # objective: current level only
        if k == 1:
            prob += t1
        else:
            prob += lam[k] + (1.0 / max(1, P)) * pulp.lpSum(mu[k][p] for p in pairs)

        _ = prob.solve(_SOLVER)
        best_layers.append(float((t1 if k == 1 else lam[k]).value() or 0.0))

        # read current x
        if required_count is None:
            x_star_last = _pick_integral_from_relaxed(x, single_group)
        else:
            xv = np.array([(v.value() or 0.0) for v in x])
            x_star_last = [j for j in range(N) if xv[j] > 0.5]
            if len(x_star_last) != int(required_count):
                x_star_last = list(np.argsort(-xv)[: int(required_count)])

        if best_layers[-1] <= 1e-8:
            break

    return x_star_last, best_layers
