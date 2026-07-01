import yaml
import numpy as np
import pulp


# --- CBC solver: no time/gap limits, quiet log ---
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
    Missing EC will be filled with 1.0.
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

    qys = [
        v["quantum_yield"]
        for v in dye_db.values()
        if v.get("quantum_yield") is not None
    ]
    mean_qy = float(np.mean(qys)) if len(qys) else 1.0

    for v in dye_db.values():
        if v.get("quantum_yield") is None:
            v["quantum_yield"] = mean_qy
        if v.get("extinction_coeff") is None:
            v["extinction_coeff"] = 1.0

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
    E = np.asarray(E, dtype=float)
    denom = np.linalg.norm(E, axis=0, keepdims=True) + 1e-12
    return E / denom


def cosine_similarity_matrix(E):
    """Cosine similarity among columns of E. Diagonal is set to 0."""
    E = np.asarray(E, dtype=float)
    norms = np.linalg.norm(E, axis=0) + 1e-12
    G = (E.T @ E) / np.outer(norms, norms)
    G = np.nan_to_num(G, nan=0.0, posinf=1.0, neginf=0.0)
    G = np.clip(G, -1.0, 1.0)
    np.fill_diagonal(G, 0.0)
    return G


def spectral_overlap_similarity_matrix(E):
    """
    Area-normalized spectral overlap among columns of E.

    Each spectrum is clipped to nonnegative values and normalized to unit area.
    Similarity is sum(min(p_i, p_j)). Range: 0 to 1. Larger means more overlap.
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

    Raw Pearson correlation is mapped from [-1, 1] to [0, 1], so larger always
    means more similar/worse.
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

    angle = arccos(cosine), similarity = 1 - angle / (pi/2).
    For nonnegative spectra this gives a score in approximately [0, 1].
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

    All supported metrics are scaled so larger values mean more similar and
    therefore worse for fluorophore separation.
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
    Given an NxN similarity matrix S, return top-k pairs.
    Each item: (value, label_i, label_j), sorted descending by value.
    """
    S = np.asarray(S, dtype=float)
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
    y = np.asarray(y, dtype=float)
    if x <= w[0]:
        return float(y[0])
    if x >= w[-1]:
        return float(y[-1])
    i = int(np.searchsorted(w, x)) - 1
    t = (x - w[i]) / (w[i + 1] - w[i])
    return float(y[i] * (1 - t) + y[i + 1] * t)


def derive_powers_simultaneous(wl, dye_db, selection_labels, laser_wavelengths):
    """Calibrate laser powers in simultaneous mode."""
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
    """Separate-mode laser power calibration."""
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
    Build effective spectra for all candidates.

    Returns:
    - E_raw
    - E_norm
    - labels_pair
    - idx_groups
    """
    W = len(wl)
    lam = np.array(sorted(laser_wavelengths), dtype=float)
    pw = np.array(powers, dtype=float)

    if pw.size < lam.size:
        pad = np.ones(lam.size - pw.size, dtype=float)
        pw = np.concatenate([pw, pad])

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
                    per_laser_blocks.append(np.asarray(em, dtype=float) * k)

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
        E_norm = _safe_l2norm_cols(E_raw)
        return E_raw, E_norm, labels, idx_groups

    # Simultaneous mode: per-segment cumulative coefficient.
    segs = _segments_from_lasers(wl, lam)

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
                loi = _nearest_idx_from_grid(wl, lo)
                hii = _nearest_idx_from_grid(wl, hi - 1) + 1
                if loi >= hii:
                    continue

                total_k = 0.0
                for m in range(i + 1):
                    ex_l = _interp_at(wl, ex, lam[m])
                    total_k += ex_l * qy * (ec if ec is not None else 1.0) * pw[m]

                eff[loi:hii] += np.asarray(em, dtype=float)[loi:hii] * total_k

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
    E_norm = _safe_l2norm_cols(E_raw)
    return E_raw, E_norm, labels, idx_groups


# =================== Global-unique constraint ===================
def _unique_dye_constraints(prob, x_vars, labels_pair):
    """Enforce each fluorophore can be used at most once globally."""
    dye_to_cols = {}
    for j, label in enumerate(labels_pair):
        d = label.split(" – ", 1)[1] if " – " in label else label
        dye_to_cols.setdefault(d, []).append(j)

    for d, cols in dye_to_cols.items():
        if len(cols) > 1:
            prob += pulp.lpSum(x_vars[j] for j in cols) <= 1, f"Unique_{d}"


# =================== Optimization helpers ===================
def _pick_integral_from_solution(x_vars, idx_groups=None, required_count=None):
    xvals = np.array([(v.value() or 0.0) for v in x_vars], dtype=float)

    if required_count is not None:
        chosen = [j for j, v in enumerate(xvals) if v > 0.5]
        if len(chosen) != int(required_count):
            chosen = list(np.argsort(-xvals)[: int(required_count)])
        return [int(j) for j in chosen]

    sel = []
    for idxs in idx_groups or []:
        if not idxs:
            continue
        j_local = int(np.argmax(xvals[idxs]))
        sel.append(int(idxs[j_local]))
    return sel


def _normalize_fixed_allowed(N, fixed_indices=None, allowed_indices=None):
    """
    Convert fixed/allowed indices to validated sets.

    allowed_indices restricts additional candidates. Fixed columns are allowed
    even if they are not listed in allowed_indices.
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


def _build_selection_model(
    C,
    idx_groups,
    labels_pair,
    enforce_unique=True,
    required_count=None,
    fixed_indices=None,
    allowed_indices=None,
    objective_coeffs=None,
    candidate_penalties=None,
    soft_penalty_weight=0.0,
    max_score_bound=None,
):
    """Build a binary selection model with linked pair variables."""
    N = C.shape[0]
    fixed, allowed, selectable = _normalize_fixed_allowed(N, fixed_indices, allowed_indices)

    if required_count is not None:
        required_count = int(required_count)
        if len(fixed) > required_count:
            raise ValueError(
                f"Number of fixed fluorophores ({len(fixed)}) cannot exceed "
                f"required_count ({required_count})."
            )
        if len(selectable) < required_count:
            raise ValueError(
                "Not enough allowed fluorophores to satisfy the requested panel size."
            )

    prob = pulp.LpProblem("fluoroselect", pulp.LpMinimize)

    x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Binary") for j in range(N)]
    y = {}

    for i in range(N):
        for j in range(i + 1, N):
            y[(i, j)] = pulp.LpVariable(f"y_{i}_{j}", lowBound=0, upBound=1)

    _add_fixed_allowed_constraints(prob, x, N, fixed_indices, allowed_indices)

    if required_count is None:
        for g, idxs in enumerate(idx_groups):
            prob += pulp.lpSum(x[j] for j in idxs) == 1, f"OnePerGroup_{g}"
        if enforce_unique:
            _unique_dye_constraints(prob, x, labels_pair)
    else:
        prob += pulp.lpSum(x) == required_count, "PickN"

    for (i, j), yij in y.items():
        prob += yij <= x[i]
        prob += yij <= x[j]
        prob += yij >= x[i] + x[j] - 1
        if max_score_bound is not None:
            prob += float(C[i, j]) * yij <= float(max_score_bound) + 1e-9

    if objective_coeffs is None:
        t = pulp.LpVariable("t", lowBound=0)
        for (i, j), yij in y.items():
            prob += t >= float(C[i, j]) * yij
        prob += t
        return prob, x, y, t

    pair_objective = pulp.lpSum(
        float(objective_coeffs[i, j]) * yij
        for (i, j), yij in y.items()
    )

    penalty_objective = 0
    if candidate_penalties is not None and float(soft_penalty_weight or 0.0) > 0:
        penalties = np.asarray(candidate_penalties, dtype=float)
        if penalties.size != N:
            raise ValueError(
                "candidate_penalties must have one value per candidate column."
            )
        penalty_objective = float(soft_penalty_weight) * pulp.lpSum(
            float(penalties[j]) * x[j]
            for j in range(N)
        )

    prob += pair_objective + penalty_objective
    return prob, x, y, None


def _solve_model(prob, x, idx_groups=None, required_count=None):
    status = prob.solve(_SOLVER)
    status_name = pulp.LpStatus.get(status, str(status))

    if status_name not in {"Optimal", "Feasible"}:
        raise ValueError(f"Optimization failed: {status_name}.")

    return _pick_integral_from_solution(
        x,
        idx_groups=idx_groups,
        required_count=required_count,
    )


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
    candidate_penalties=None,
    soft_penalty_weight=0.0,
):
    """
    Minimize the maximum pairwise similarity among selected columns.
    """
    N = E_norm.shape[1]
    if N == 0:
        return [], 0.0

    C = similarity_matrix(E_norm, metric=similarity_metric)

    prob, x, _, t = _build_selection_model(
        C,
        idx_groups,
        labels_pair,
        enforce_unique=enforce_unique,
        required_count=required_count,
        fixed_indices=fixed_indices,
        allowed_indices=allowed_indices,
        objective_coeffs=None,
        max_score_bound=None,
    )

    x_star = _solve_model(prob, x, idx_groups=idx_groups, required_count=required_count)
    t_val = float(t.value() or 0.0)
    return x_star, t_val


# =================== Optimization: lexicographic-like ===================
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
    candidate_penalties=None,
    soft_penalty_weight=0.0,
):
    """
    Select fluorophores by first minimizing the worst selected-pair score,
    then minimizing a top-heavy weighted sum while keeping that worst score fixed.

    This preserves the minimax optimum and reduces secondary high-overlap pairs.
    If candidate_penalties are provided, lower-priority fluorophores are avoided
    in the second stage without sacrificing the best worst-pair score.
    """
    N = E_norm.shape[1]
    if N == 0:
        return [], 0.0

    C = similarity_matrix(E_norm, metric=similarity_metric)

    sel0, best_t = solve_minimax_layer(
        E_norm,
        idx_groups,
        labels_pair,
        enforce_unique=enforce_unique,
        required_count=required_count,
        fixed_indices=fixed_indices,
        allowed_indices=allowed_indices,
        similarity_metric=similarity_metric,
    )

    # Second stage: among all minimax-optimal panels, reduce the remaining high scores.
    # Power weights emphasize high-overlap pairs without requiring order-statistic variables.
    power = max(2, int(levels))
    coeffs = np.power(np.maximum(C, 0.0) + 1e-12, power)

    try:
        prob, x, _, _ = _build_selection_model(
            C,
            idx_groups,
            labels_pair,
            enforce_unique=enforce_unique,
            required_count=required_count,
            fixed_indices=fixed_indices,
            allowed_indices=allowed_indices,
            objective_coeffs=coeffs,
            candidate_penalties=candidate_penalties,
            soft_penalty_weight=soft_penalty_weight,
            max_score_bound=best_t,
        )
        sel = _solve_model(prob, x, idx_groups=idx_groups, required_count=required_count)
    except Exception:
        sel = sel0

    return sel, float(best_t)
