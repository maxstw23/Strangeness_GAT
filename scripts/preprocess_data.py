import uproot
import awkward as ak
import numpy as np
import torch
from tqdm import tqdm
import os
import config

# PDG masses in GeV/c²
M_KAON  = 0.493677
M_OMEGA = 1.67245


def compute_kstar(k_px, k_py, k_pz, o_px, o_py, o_pz):
    """Lorentz-invariant relative momentum in the kaon-omega pair rest frame."""
    E_k = np.sqrt(k_px**2 + k_py**2 + k_pz**2 + M_KAON**2)
    E_o = np.sqrt(float(o_px)**2 + float(o_py)**2 + float(o_pz)**2 + M_OMEGA**2)
    s = (E_k + E_o)**2 - (k_px + o_px)**2 - (k_py + o_py)**2 - (k_pz + o_pz)**2
    kallen = (s - (M_KAON + M_OMEGA)**2) * (s - (M_KAON - M_OMEGA)**2)
    return np.sqrt(np.maximum(kallen, 0.0)) / (2.0 * np.sqrt(np.maximum(s, 1e-10)))


def compute_cos_theta_star(k_px, k_py, k_pz, o_px, o_py, o_pz):
    """Beam-axis angle in the pair rest frame: cos(θ*) = k*_z / |k*|.
    k*_z is the z-component of the kaon momentum after boosting to the kaon-Omega
    pair rest frame. θ* is the angle between k* and the beam axis (z-hat).
    Motivated by source elongation along z: R_long > R_out ~ R_side.
    """
    E_k = np.sqrt(k_px**2 + k_py**2 + k_pz**2 + M_KAON**2)
    E_o = np.sqrt(float(o_px)**2 + float(o_py)**2 + float(o_pz)**2 + M_OMEGA**2)

    # Pair 4-momentum
    pair_E  = E_k + E_o
    pair_px = k_px + float(o_px)
    pair_py = k_py + float(o_py)
    pair_pz = k_pz + float(o_pz)
    pair_M  = np.sqrt(np.maximum(pair_E**2 - pair_px**2 - pair_py**2 - pair_pz**2, 1e-10))

    # Boost vector β = p_pair / E_pair, γ = E_pair / M_pair
    beta_x = pair_px / np.maximum(pair_E, 1e-10)
    beta_y = pair_py / np.maximum(pair_E, 1e-10)
    beta_z = pair_pz / np.maximum(pair_E, 1e-10)
    beta2  = beta_x**2 + beta_y**2 + beta_z**2
    gamma  = pair_E / pair_M

    # Lorentz-boost kaon to pair rest frame: p* = p + [(γ−1)/β²](β·p)β − γβE
    beta_dot_p = beta_x * k_px + beta_y * k_py + beta_z * k_pz
    factor     = (gamma - 1.0) / np.maximum(beta2, 1e-10)
    kstar_x = k_px + factor * beta_dot_p * beta_x - gamma * beta_x * E_k
    kstar_y = k_py + factor * beta_dot_p * beta_y - gamma * beta_y * E_k
    kstar_z = k_pz + factor * beta_dot_p * beta_z - gamma * beta_z * E_k

    k_p_star = np.sqrt(kstar_x**2 + kstar_y**2 + kstar_z**2)
    return np.clip(kstar_z / np.maximum(k_p_star, 1e-10), -1.0, 1.0)


def compute_delta_y(k_px, k_py, k_pz, o_px, o_py, o_pz):
    """Rapidity difference: y_kaon − y_Omega (boost-invariant, uses PDG masses)."""
    E_k = np.sqrt(k_px**2 + k_py**2 + k_pz**2 + M_KAON**2)
    E_o = np.sqrt(float(o_px)**2 + float(o_py)**2 + float(o_pz)**2 + M_OMEGA**2)
    y_k = 0.5 * np.log(np.maximum(E_k + k_pz, 1e-10) / np.maximum(E_k - k_pz, 1e-10))
    y_o = 0.5 * np.log(np.maximum(E_o + float(o_pz), 1e-10) / np.maximum(E_o - float(o_pz), 1e-10))
    return y_k - y_o


def compute_omega_rapidity(o_px, o_py, o_pz):
    """Omega rapidity (signed)."""
    E_o = np.sqrt(float(o_px)**2 + float(o_py)**2 + float(o_pz)**2 + M_OMEGA**2)
    return 0.5 * np.log(
        np.maximum(E_o + float(o_pz), 1e-10) / np.maximum(E_o - float(o_pz), 1e-10)
    )


def run_balanced_preprocessing(input_file, output_file):
    print(f"Reading STAR TTree from {input_file}...")
    with uproot.open(input_file) as file:
        tree = file["ml_tree"]
        data = tree.arrays([
            "omega_px", "omega_py", "omega_pz", "omega_charge",
            "kaon_px", "kaon_py", "kaon_pz", "kaon_charge",
            "EPDEP_1st", "EPDEP_2nd"
        ], library="ak")

    # Build event-mixed K⁻ pool: bin Anti events by (|y_Ω|, pT_Ω) quartiles so
    # fake kaons drawn for padding have realistic kinematic relationships to their Omega.
    Y_BINS  = [0.0, 0.24, 0.49, 0.78, np.inf]   # |y_Ω| quartiles
    PT_BINS = [0.0, 1.41, 1.75, 2.14, np.inf]   # pT_Ω quartiles
    MIN_POOL_SIZE = 50

    print("Building event-mixed K⁻ pool for Anti events...")
    pool_lists = {}
    for i in range(len(data)):
        if int(data[i]["omega_charge"]) <= 0:
            continue  # only Anti (Ω̄⁺) events contribute to the pool
        o_px_i, o_py_i, o_pz_i = data[i]["omega_px"], data[i]["omega_py"], data[i]["omega_pz"]
        y_o_i  = abs(float(compute_omega_rapidity(o_px_i, o_py_i, o_pz_i)))
        pt_o_i = float(np.sqrt(float(o_px_i)**2 + float(o_py_i)**2))
        yb  = int(np.searchsorted(Y_BINS[1:],  y_o_i))
        ptb = int(np.searchsorted(PT_BINS[1:], pt_o_i))
        k_q_i  = ak.to_numpy(data[i]["kaon_charge"])
        mask_i = k_q_i < 0
        kpx_i  = ak.to_numpy(data[i]["kaon_px"])[mask_i]
        kpy_i  = ak.to_numpy(data[i]["kaon_py"])[mask_i]
        kpz_i  = ak.to_numpy(data[i]["kaon_pz"])[mask_i]
        if len(kpx_i) == 0:
            continue
        pool_lists.setdefault((yb, ptb), []).append(
            np.stack([kpx_i, kpy_i, kpz_i], axis=1))

    mixed_pool    = {k: np.concatenate(v, axis=0) for k, v in pool_lists.items()}
    global_fallback = np.concatenate(list(mixed_pool.values()), axis=0)

    processed_graphs = []

    print(f"Processing {len(data)} events...")
    for i in tqdm(range(len(data))):
        k_px = ak.to_numpy(data[i]["kaon_px"])
        k_py = ak.to_numpy(data[i]["kaon_py"])
        k_pz = ak.to_numpy(data[i]["kaon_pz"])
        k_q = ak.to_numpy(data[i]["kaon_charge"])

        o_px, o_py, o_pz = data[i]["omega_px"], data[i]["omega_py"], data[i]["omega_pz"]
        o_charge = data[i]["omega_charge"]

        # Keep only opposite-sign kaons (the strangeness-balancing partners)
        # Ω⁻ (o_charge < 0): opposite-sign = K⁺
        # Ω̄⁺ (o_charge > 0): opposite-sign = K⁻
        if o_charge < 0:
            oppo_mask = k_q > 0
        else:
            oppo_mask = k_q < 0

        n_pos = np.sum(k_q > 0)   # K⁺ count — target multiplicity (K⁺ > K⁻ globally)
        if n_pos == 0: continue

        f_px = k_px[oppo_mask]
        f_py = k_py[oppo_mask]
        f_pz = k_pz[oppo_mask]
        f_q = k_q[oppo_mask]  # kaon charges for selected kaons
        n_oppo = len(f_px)

        # For Ω̄⁺ events: pad K⁻ to reach K⁺ count using event-mixed pool (or trim if K⁻ > K⁺)
        if o_charge > 0:
            y_o_val  = abs(float(compute_omega_rapidity(o_px, o_py, o_pz)))
            pt_o_val = float(np.sqrt(float(o_px)**2 + float(o_py)**2))
            yb  = int(np.searchsorted(Y_BINS[1:],  y_o_val))
            ptb = int(np.searchsorted(PT_BINS[1:], pt_o_val))
            pool = mixed_pool.get((yb, ptb), global_fallback)
            if len(pool) < MIN_POOL_SIZE:
                pool = global_fallback
            if n_oppo < n_pos:
                idx = np.random.choice(len(pool), size=n_pos - n_oppo, replace=True)
                samples = pool[idx]
                f_px = np.concatenate([f_px, samples[:, 0]])
                f_py = np.concatenate([f_py, samples[:, 1]])
                f_pz = np.concatenate([f_pz, samples[:, 2]])
                f_q = np.concatenate([f_q, np.full(len(samples), -1)])  # padded kaons are K⁻
            elif n_oppo > n_pos:
                idx = np.random.choice(n_oppo, size=n_pos, replace=False)
                f_px, f_py, f_pz = f_px[idx], f_py[idx], f_pz[idx]
                f_q = f_q[idx]

        # --- COORDINATE TRANSFORMATION ---
        # 1. Omega Kinematics
        o_pt = np.sqrt(o_px ** 2 + o_py ** 2)
        o_p = np.sqrt(o_px ** 2 + o_py ** 2 + o_pz ** 2)
        o_eta = np.arctanh(o_pz / o_p)
        o_phi = np.arctan2(o_py, o_px)

        # 2. Kaon Kinematics
        f_pt = np.sqrt(f_px ** 2 + f_py ** 2)
        f_p = np.sqrt(f_px ** 2 + f_py ** 2 + f_pz ** 2)
        f_eta = np.arctanh(f_pz / f_p)
        f_phi = np.arctan2(f_py, f_px)

        # 3. Relative Features
        k_star       = compute_kstar(f_px, f_py, f_pz, o_px, o_py, o_pz)
        cos_theta_st = compute_cos_theta_star(f_px, f_py, f_pz, o_px, o_py, o_pz)
        # |d_y|: absolute rapidity difference — Au+Au is forward-backward symmetric
        delta_y = compute_delta_y(f_px, f_py, f_pz, o_px, o_py, o_pz)
        d_y  = np.abs(delta_y)
        # d_y_signed = |y_K| − |y_Ω|: midrapidity proximity gap
        # Negative → kaon closer to midrapidity than Omega (expected for junction fragmentation)
        y_o = compute_omega_rapidity(o_px, o_py, o_pz)
        y_k = delta_y + y_o
        d_y_signed = np.abs(y_k) - abs(y_o)
        # |d_phi|: absolute azimuthal separation in [0, π] — no preferred azimuthal direction
        d_phi = np.abs(((f_phi - o_phi) + np.pi) % (2 * np.pi) - np.pi)

        # Event-plane angles (one scalar per event)
        psi1 = float(data[i]["EPDEP_1st"])
        psi2 = float(data[i]["EPDEP_2nd"])

        # Broadcast (Omega azimuthal alignment with event plane)
        o_cos_psi1  = np.full(len(f_px), np.cos(o_phi - psi1))
        o_cos2_psi2 = np.full(len(f_px), np.cos(2.0 * (o_phi - psi2)))

        # Per-kaon (kaon azimuthal alignment with event plane)
        f_cos_psi1  = np.cos(f_phi - psi1)
        f_cos2_psi2 = np.cos(2.0 * (f_phi - psi2))

        net_kaon_val = int(np.sum(k_q > 0)) - int(np.sum(k_q < 0))
        o_pt_broadcast = np.full(len(f_px), o_pt)
        o_y_abs = np.full(len(f_px), abs(y_o))  # index 11: |y_Ω| broadcast
        net_kaon_broadcast = np.full(len(f_px), net_kaon_val)
        # Per-kaon 1/ε: low efficiency → higher weight (kaon represents more true kaons)
        eff_weights = np.array([
            config.get_inv_eff(float(pt), float(eta), cent=8, charge=int(q))
            for pt, eta, q in zip(f_pt, f_eta, f_q)
        ])
        node_features = np.stack([
            f_pt, k_star, d_y, d_phi, o_pt_broadcast, cos_theta_st,
            o_cos_psi1, o_cos2_psi2, f_cos_psi1, f_cos2_psi2,
            d_y_signed, o_y_abs, net_kaon_broadcast, eff_weights  # indices 10–13
        ], axis=1)

        y_label = 1 if o_charge > 0 else 0
        processed_graphs.append({
            'x': torch.tensor(node_features, dtype=torch.float),
            'y': torch.tensor([y_label], dtype=torch.long)
        })

    print(f"Saving {len(processed_graphs)} graphs...")
    torch.save(processed_graphs, output_file)

    print("Computing feature statistics...")
    all_features = torch.cat([g['x'] for g in processed_graphs], dim=0)
    means = all_features.mean(dim=0)
    stds = all_features.std(dim=0)
    stds[stds == 0] = 1.0  # avoid division by zero for constant features (e.g. f_q after balancing)
    stats_file = output_file.replace(".pt", "_stats.pt")
    torch.save({'means': means, 'stds': stds}, stats_file)
    print(f"Feature stats saved to {stats_file}")
    feature_names = [
        "f_pt", "k_star", "d_y", "d_phi", "o_pt", "cos_theta_star",
        "o_cos_psi1", "o_cos2_psi2", "f_cos_psi1", "f_cos2_psi2",
        "d_y_signed", "o_y_abs", "net_kaon", "eff_weight"
    ]
    for name, m, s in zip(feature_names, means, stds):
        print(f"  {name}: mean={m:.4f}, std={s:.4f}")
    print("Done!")


def run_unpadded_preprocessing(input_file, output_file):
    """Like run_balanced_preprocessing but without K⁻ padding.

    On unpadded data n_kaons(Ω⁻) ≈ n_{K+} > n_{K-} ≈ n_kaons(Ω̄⁺), preserving the
    genuine K⁺/K⁻ asymmetry so a GRL adversary can do real work removing it.

    Feature layout: identical 13-column layout to balanced dataset, so FEATURE_REGISTRY
    indices work for both datasets without any special casing.
      index  2 — d_y      = |y_K − y_Ω|  [absolute rapidity gap]
      index 10 — d_y_signed = |y_K| − |y_Ω|  [midrapidity proximity gap]
      index 11 — o_y_abs  = |y_Ω|  [Omega rapidity displacement]
      index 12 — net_kaon = n_K+(event) − n_K−(event)  [event charge asymmetry]
    """
    print(f"Reading STAR TTree from {input_file}...")
    with uproot.open(input_file) as file:
        tree = file["ml_tree"]
        data = tree.arrays([
            "omega_px", "omega_py", "omega_pz", "omega_charge",
            "kaon_px", "kaon_py", "kaon_pz", "kaon_charge",
            "EPDEP_1st", "EPDEP_2nd"
        ], library="ak")

    processed_graphs = []

    print(f"Processing {len(data)} events (unpadded)...")
    for i in tqdm(range(len(data))):
        k_px = ak.to_numpy(data[i]["kaon_px"])
        k_py = ak.to_numpy(data[i]["kaon_py"])
        k_pz = ak.to_numpy(data[i]["kaon_pz"])
        k_q = ak.to_numpy(data[i]["kaon_charge"])

        o_px, o_py, o_pz = data[i]["omega_px"], data[i]["omega_py"], data[i]["omega_pz"]
        o_charge = data[i]["omega_charge"]

        # Keep only opposite-sign kaons; no padding
        oppo_mask = k_q > 0 if o_charge < 0 else k_q < 0
        f_px = k_px[oppo_mask]
        f_py = k_py[oppo_mask]
        f_pz = k_pz[oppo_mask]
        if len(f_px) == 0:
            continue

        # --- COORDINATE TRANSFORMATION ---
        o_pt = np.sqrt(o_px ** 2 + o_py ** 2)
        o_phi = np.arctan2(o_py, o_px)

        f_pt  = np.sqrt(f_px ** 2 + f_py ** 2)
        f_p   = np.sqrt(f_px ** 2 + f_py ** 2 + f_pz ** 2)
        f_eta = np.arctanh(np.clip(f_pz / np.maximum(f_p, 1e-10), -1 + 1e-7, 1 - 1e-7))
        f_phi = np.arctan2(f_py, f_px)

        k_star       = compute_kstar(f_px, f_py, f_pz, o_px, o_py, o_pz)
        cos_theta_st = compute_cos_theta_star(f_px, f_py, f_pz, o_px, o_py, o_pz)

        # Absolute rapidity gap (col 2, same as balanced)
        delta_y = compute_delta_y(f_px, f_py, f_pz, o_px, o_py, o_pz)
        d_y = np.abs(delta_y)

        # Directed rapidity gap: |y_K| − |y_Ω| (col 10)
        # Negative → kaon closer to midrapidity than Omega (junction expectation)
        # NOTE: sign(y_Ω)×(y_K−y_Ω) is WRONG — false negative when y_K crosses zero.
        y_o = compute_omega_rapidity(o_px, o_py, o_pz)
        y_k = delta_y + y_o
        d_y_signed = np.abs(y_k) - abs(y_o)

        d_phi = np.abs(((f_phi - o_phi) + np.pi) % (2 * np.pi) - np.pi)

        psi1 = float(data[i]["EPDEP_1st"])
        psi2 = float(data[i]["EPDEP_2nd"])

        o_pt_broadcast = np.full(len(f_px), o_pt)
        o_cos_psi1  = np.full(len(f_px), np.cos(o_phi - psi1))
        o_cos2_psi2 = np.full(len(f_px), np.cos(2.0 * (o_phi - psi2)))
        f_cos_psi1  = np.cos(f_phi - psi1)
        f_cos2_psi2 = np.cos(2.0 * (f_phi - psi2))

        o_y_abs = np.full(len(f_px), abs(y_o))
        net_kaon_val = int(np.sum(k_q > 0)) - int(np.sum(k_q < 0))
        net_kaon_broadcast = np.full(len(f_px), net_kaon_val)

        # Per-kaon 1/ε: low efficiency → higher weight (kaon represents more true kaons)
        f_charge = 1 if int(o_charge) < 0 else -1  # K+ for Ω⁻, K- for Ω̄⁺
        eff_weights = np.array([
            config.get_inv_eff(float(pt), float(eta), cent=8, charge=f_charge)
            for pt, eta in zip(f_pt, f_eta)
        ])

        # 14 features: 13-column layout + eff_weight at index 13
        node_features = np.stack([
            f_pt, k_star, d_y, d_phi, o_pt_broadcast, cos_theta_st,
            o_cos_psi1, o_cos2_psi2, f_cos_psi1, f_cos2_psi2,
            d_y_signed, o_y_abs, net_kaon_broadcast, eff_weights  # indices 10–13
        ], axis=1)

        y_label = 1 if o_charge > 0 else 0
        processed_graphs.append({
            'x': torch.tensor(node_features, dtype=torch.float),
            'y': torch.tensor([y_label], dtype=torch.long)
        })

    print(f"Saving {len(processed_graphs)} graphs...")
    torch.save(processed_graphs, output_file)

    print("Computing feature statistics...")
    all_features = torch.cat([g['x'] for g in processed_graphs], dim=0)
    means = all_features.mean(dim=0)
    stds = all_features.std(dim=0)
    stds[stds == 0] = 1.0
    stats_file = output_file.replace(".pt", "_stats.pt")
    torch.save({'means': means, 'stds': stds}, stats_file)
    print(f"Feature stats saved to {stats_file}")
    feature_names = [
        "f_pt", "k_star", "d_y", "d_phi", "o_pt", "cos_theta_star",
        "o_cos_psi1", "o_cos2_psi2", "f_cos_psi1", "f_cos2_psi2",
        "d_y_signed", "o_y_abs", "net_kaon", "eff_weight"
    ]
    for name, m, s in zip(feature_names, means, stds):
        print(f"  {name}: mean={m:.4f}, std={s:.4f}")
    print("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["padded", "unpadded"], default="padded",
                        help="padded: balanced K⁻ padding (default); unpadded: no padding, directed rapidity features")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    if args.mode == "unpadded":
        run_unpadded_preprocessing(
            input_file="data/omega_kaon_all.root",
            output_file="data/unpadded_omega_anti.pt"
        )
    else:
        run_balanced_preprocessing(
            input_file="data/omega_kaon_all.root",
            output_file="data/balanced_omega_anti.pt"
        )