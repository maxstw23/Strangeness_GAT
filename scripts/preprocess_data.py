import uproot
import awkward as ak
import numpy as np
import torch
from tqdm import tqdm
import os

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

    print("Building global K⁻ momentum pool for balancing...")
    all_k_px = ak.to_numpy(ak.flatten(data["kaon_px"]))
    all_k_py = ak.to_numpy(ak.flatten(data["kaon_py"]))
    all_k_pz = ak.to_numpy(ak.flatten(data["kaon_pz"]))
    all_k_q = ak.to_numpy(ak.flatten(data["kaon_charge"]))

    k_neg_pool = np.stack([all_k_px[all_k_q < 0], all_k_py[all_k_q < 0], all_k_pz[all_k_q < 0]], axis=1)

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
        n_oppo = len(f_px)

        # For Ω̄⁺ events: pad K⁻ to reach K⁺ count (or trim if K⁻ > K⁺, rare)
        if o_charge > 0:
            if n_oppo < n_pos:
                idx = np.random.choice(len(k_neg_pool), size=n_pos - n_oppo, replace=True)
                samples = k_neg_pool[idx]
                f_px = np.concatenate([f_px, samples[:, 0]])
                f_py = np.concatenate([f_py, samples[:, 1]])
                f_pz = np.concatenate([f_pz, samples[:, 2]])
            elif n_oppo > n_pos:
                idx = np.random.choice(n_oppo, size=n_pos, replace=False)
                f_px, f_py, f_pz = f_px[idx], f_py[idx], f_pz[idx]

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

        o_pt_broadcast = np.full(len(f_px), o_pt)
        o_y_abs = np.full(len(f_px), abs(y_o))  # index 11: |y_Ω| broadcast
        node_features = np.stack([
            f_pt, k_star, d_y, d_phi, o_pt_broadcast, cos_theta_st,
            o_cos_psi1, o_cos2_psi2, f_cos_psi1, f_cos2_psi2,
            d_y_signed, o_y_abs  # index 10, 11
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
        "d_y_signed", "o_y_abs"
    ]
    for name, m, s in zip(feature_names, means, stds):
        print(f"  {name}: mean={m:.4f}, std={s:.4f}")
    print("Done!")


def run_unpadded_preprocessing(input_file, output_file):
    """Like run_balanced_preprocessing but without K⁻ padding.

    On unpadded data n_kaons(Ω⁻) ≈ n_{K+} > n_{K-} ≈ n_kaons(Ω̄⁺), preserving the
    genuine K⁺/K⁻ asymmetry so a GRL adversary can do real work removing it.

    Feature differences vs balanced:
      index 2 — d_y_signed = |y_K| − |y_Omega|  [midrapidity proximity gap;
                 negative means kaon is closer to midrapidity than the Omega, as expected
                 for junction-transport (string fragmentation between beam and Omega);
                 near zero / positive for pair-produced]
      index 10 — o_y_abs = |y_Omega|  [Omega displacement from midrapidity, broadcast;
                  larger for junction Ω⁻ (beam-remnant origin) than thermal Ω̄⁺]
    Total: 11 features per kaon (vs 10 in balanced dataset).
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

        f_pt = np.sqrt(f_px ** 2 + f_py ** 2)
        f_phi = np.arctan2(f_py, f_px)

        k_star       = compute_kstar(f_px, f_py, f_pz, o_px, o_py, o_pz)
        cos_theta_st = compute_cos_theta_star(f_px, f_py, f_pz, o_px, o_py, o_pz)

        # Directed rapidity gap: |y_K| − |y_Ω|
        # Negative → kaon is closer to midrapidity than Omega (expected for junction: string
        #   fragments between beam remnant and Omega deposit kaons at smaller |y|)
        # Positive → kaon is further from midrapidity than Omega
        # Near zero → symmetric (expected for pair-produced)
        # NOTE: sign(y_Ω)×(y_K−y_Ω) is WRONG — it gives a false negative when y_K
        # crosses zero (kaon on the opposite side of midrapidity from Omega, yet further away).
        y_o = compute_omega_rapidity(o_px, o_py, o_pz)
        y_k = compute_delta_y(f_px, f_py, f_pz, o_px, o_py, o_pz) + y_o  # y_k = delta_y + y_o
        d_y_signed = np.abs(y_k) - abs(y_o)

        d_phi = np.abs(((f_phi - o_phi) + np.pi) % (2 * np.pi) - np.pi)

        psi1 = float(data[i]["EPDEP_1st"])
        psi2 = float(data[i]["EPDEP_2nd"])

        o_pt_broadcast = np.full(len(f_px), o_pt)
        o_cos_psi1  = np.full(len(f_px), np.cos(o_phi - psi1))
        o_cos2_psi2 = np.full(len(f_px), np.cos(2.0 * (o_phi - psi2)))
        f_cos_psi1  = np.cos(f_phi - psi1)
        f_cos2_psi2 = np.cos(2.0 * (f_phi - psi2))

        # |y_Omega|: Omega's displacement from midrapidity (broadcast)
        o_y_abs = np.full(len(f_px), abs(y_o))

        # 11 features: indices 0-9 match balanced layout (d_y at index 2 is now directed),
        # index 10 = o_y_abs (new broadcast feature, unpadded dataset only)
        node_features = np.stack([
            f_pt, k_star, d_y_signed, d_phi, o_pt_broadcast, cos_theta_st,
            o_cos_psi1, o_cos2_psi2, f_cos_psi1, f_cos2_psi2,
            o_y_abs
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
        "f_pt", "k_star", "d_y_signed", "d_phi", "o_pt", "cos_theta_star",
        "o_cos_psi1", "o_cos2_psi2", "f_cos_psi1", "f_cos2_psi2", "o_y_abs"
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