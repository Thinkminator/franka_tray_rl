#!/usr/bin/env python3
"""
Systematic PD gain search for Panda arm holding pose under tray load.
Tests combinations of Kq multipliers and reports best stable configuration.
"""
import os
import sys
import numpy as np
import mujoco

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.traypose.traypose_env import TrayPoseEnv


def evaluate_gains(env, Kq, Dq, hold_seconds=2.0, verbose=False):
    """
    Test given gains by holding start pose with zero action.
    Returns: (max_pos_error_rad, max_vel, saturation_fraction, tray_drop_cm)
    """
    # Backup current gains
    Kq_backup, Dq_backup = env.Kq.copy(), env.Dq.copy()
    env.Kq, env.Dq = Kq.copy(), Dq.copy()
    
    # Reset to start pose
    env.reset()
    q_ref = env._get_arm_qpos().copy()
    tray_z_start = env.tray_pos[2]
    
    n_steps = int(hold_seconds / env.control_dt)
    max_pos_err = 0.0
    max_vel = 0.0
    total_sat = 0
    total_torque_checks = 0
    
    for step in range(n_steps):
        # Zero action
        action = np.zeros(7, dtype=np.float32)
        env.step(action)
        
        # Track errors
        q_now = env._get_arm_qpos()
        qd_now = env._get_arm_qvel()
        pos_err = np.abs(q_now - q_ref)
        max_pos_err = max(max_pos_err, float(np.max(pos_err)))
        max_vel = max(max_vel, float(np.max(np.abs(qd_now))))
        
        # Check saturation (compute raw tau before clipping)
        err = env.q_des - q_now
        tau_raw = env.Kq * err - env.Dq * qd_now
        tau_raw += np.array([env.data.qfrc_bias[d] for d in env.arm_dofadr], dtype=np.float64)
        
        saturated = np.abs(tau_raw) > env.tau_limits
        total_sat += np.sum(saturated)
        total_torque_checks += 7
    
    tray_z_end = env.tray_pos[2]
    tray_drop_cm = (tray_z_start - tray_z_end) * 100.0  # convert to cm
    sat_fraction = total_sat / total_torque_checks if total_torque_checks > 0 else 0.0
    
    # Restore gains
    env.Kq, env.Dq = Kq_backup, Dq_backup
    
    if verbose:
        print(f"    max_err={max_pos_err*180/np.pi:.2f}°, max_vel={max_vel:.3f} rad/s, "
              f"sat={sat_fraction*100:.1f}%, tray_drop={tray_drop_cm:.2f} cm")
    
    return max_pos_err, max_vel, sat_fraction, tray_drop_cm


def grid_search_gains(env, hold_seconds=2.0):
    """
    Grid search over gain multipliers for proximal (1-4) and distal (5-7) joints.
    """
    base_Kq = np.array([200.0, 200.0, 200.0, 150.0, 100.0, 80.0, 60.0], dtype=np.float64)
    
    # Search ranges
    prox_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    dist_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    results = []
    
    print(f"\n{'='*80}")
    print(f"GRID SEARCH: Testing {len(prox_multipliers)} × {len(dist_multipliers)} = "
          f"{len(prox_multipliers)*len(dist_multipliers)} combinations")
    print(f"Hold duration: {hold_seconds}s")
    print(f"Base Kq: {base_Kq}")
    print(f"{'='*80}\n")
    
    total_tests = len(prox_multipliers) * len(dist_multipliers)
    test_num = 0
    
    for m_prox in prox_multipliers:
        for m_dist in dist_multipliers:
            test_num += 1
            
            # Build gains
            Kq = base_Kq.copy()
            Kq[:4] *= m_prox
            Kq[4:] *= m_dist
            Dq = 2.0 * np.sqrt(Kq)
            
            print(f"[{test_num}/{total_tests}] Testing m_prox={m_prox:.1f}, m_dist={m_dist:.1f}")
            print(f"  Kq = {Kq}")
            
            max_err, max_vel, sat_frac, tray_drop = evaluate_gains(
                env, Kq, Dq, hold_seconds=hold_seconds, verbose=True
            )
            
            results.append({
                'm_prox': m_prox,
                'm_dist': m_dist,
                'Kq': Kq.copy(),
                'Dq': Dq.copy(),
                'max_err_deg': max_err * 180 / np.pi,
                'max_vel': max_vel,
                'sat_frac': sat_frac,
                'tray_drop_cm': tray_drop,
                'score': compute_score(max_err, max_vel, sat_frac, tray_drop)
            })
    
    return results


def compute_score(max_err, max_vel, sat_frac, tray_drop_cm):
    """
    Lower is better. Penalize:
    - Position error (most important)
    - Tray drop (critical)
    - Saturation (bad for control)
    - High velocities (instability)
    """
    err_penalty = max_err * 1000.0  # rad to score
    drop_penalty = abs(tray_drop_cm) * 10.0  # cm drop
    sat_penalty = sat_frac * 50.0
    vel_penalty = max_vel * 5.0
    
    return err_penalty + drop_penalty + sat_penalty + vel_penalty


def print_top_results(results, top_n=10):
    """Print best configurations"""
    sorted_results = sorted(results, key=lambda x: x['score'])
    
    print(f"\n{'='*80}")
    print(f"TOP {top_n} CONFIGURATIONS (lower score = better)")
    print(f"{'='*80}\n")
    
    for i, res in enumerate(sorted_results[:top_n], 1):
        print(f"#{i} | Score: {res['score']:.2f}")
        print(f"    m_prox={res['m_prox']:.1f}, m_dist={res['m_dist']:.1f}")
        print(f"    Kq = {res['Kq']}")
        print(f"    Dq = {res['Dq']}")
        print(f"    Max error: {res['max_err_deg']:.2f}°")
        print(f"    Max vel: {res['max_vel']:.3f} rad/s")
        print(f"    Saturation: {res['sat_frac']*100:.1f}%")
        print(f"    Tray drop: {res['tray_drop_cm']:.2f} cm")
        print()
    
    # Print best as code snippet
    best = sorted_results[0]
    print(f"{'='*80}")
    print("RECOMMENDED GAINS (paste into __init__):")
    print(f"{'='*80}")
    print(f"self.Kq = np.array({list(best['Kq'])}, dtype=np.float64)")
    print(f"self.Dq = np.array({list(best['Dq'])}, dtype=np.float64)")
    print(f"{'='*80}\n")


def main():
    print("\n🔧 Initializing environment for gain search...")
    env = TrayPoseEnv(model_path="assets/panda_tray/panda_tray_cylinder.xml")
    
    # Run grid search
    results = grid_search_gains(env, hold_seconds=2.0)
    
    # Print results
    print_top_results(results, top_n=15)
    
    # Save to file
    import json
    output_file = "gain_search_results.json"
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON
        json_results = []
        for r in results:
            json_results.append({
                'm_prox': r['m_prox'],
                'm_dist': r['m_dist'],
                'Kq': r['Kq'].tolist(),
                'Dq': r['Dq'].tolist(),
                'max_err_deg': r['max_err_deg'],
                'max_vel': r['max_vel'],
                'sat_frac': r['sat_frac'],
                'tray_drop_cm': r['tray_drop_cm'],
                'score': r['score']
            })
        json.dump(json_results, f, indent=2)
    
    print(f"✓ Full results saved to {output_file}")
    
    env.close()


if __name__ == "__main__":
    main()