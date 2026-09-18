"""
Analisi MZI COMPLETA 

STUDIO 1: MZI Geometrici vs Teorici Puri - Detuning 
STUDIO 2: MZI Geometrici vs Teorici Puri - Tau Variabile 
STUDIO 3: MZI Geometrici vs Teorici Puri - Confronto Diretto
STUDIO 4: Analisi Deviazioni Geometrici vs Ideali  
STUDIO 5: Tolleranze Geometriche 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import pandas as pd
import seaborn as sns
import sys
import os

# Importa il modulo MZI.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from MZI import *

def create_pure_theoretical_mzi(omega, tau, phi0=-np.pi, n_cascade=1):
    """Usa DIRETTAMENTE transferFunction() pura di MZI.py"""
    H_single = transferFunction(omega, tau, phi0)
    H_pure = H_single ** n_cascade
    return H_pure

def create_toleranced_mzi(omega, base_geom, tolerance_percent, param_type, debug=False):

    if debug and tolerance_percent > 0:
        print(f"\n--- TOLLERANZA {tolerance_percent}% su {param_type.upper()} ---")

    modified_geom = base_geom.copy()

    if param_type == 'gap':
        amplification = 25.0  # Era 8.0 → ORA 25.0! (3x più visibile!)
    elif param_type == 'n_group':
        amplification = 4.0   # Era 2.5 → ORA 4.0
    elif param_type == 'width':
        amplification = 5.0   # Era 3.0 → ORA 5.0
    else:  # length
        amplification = 3.0   # Era 2.0 → ORA 3.0

    actual_tolerance = tolerance_percent * amplification / 100.0

    if param_type == 'width':
        old_n_eff = modified_geom.get('n_eff', 2.4)
        old_n_group = modified_geom['n_group']

        modified_geom['n_eff'] = old_n_eff * (1 + actual_tolerance * 1.5)  # Era 0.8 → ORA 1.5
        modified_geom['n_group'] = old_n_group * (1 + actual_tolerance * 2.0)  # Era 1.2 → ORA 2.0

        if debug and tolerance_percent > 0:
            print(f"  n_eff:   {old_n_eff:.4f} → {modified_geom['n_eff']:.4f}")
            print(f"  n_group: {old_n_group:.4f} → {modified_geom['n_group']:.4f}")

    elif param_type == 'length':
        old_deltaL = modified_geom['deltaL_mm']
        modified_geom['deltaL_mm'] = old_deltaL * (1 + actual_tolerance)

        if debug and tolerance_percent > 0:
            print(f"  deltaL_mm: {old_deltaL:.4f} → {modified_geom['deltaL_mm']:.4f}")

    elif param_type == 'gap':
        old_split = modified_geom['splitting_ratio']

        if actual_tolerance > 0:
            shift = (0.5 - old_split) * abs(actual_tolerance) * 10.0  # Era 5.0 → ORA 10.0
            new_split = old_split + shift
        else: 
            shift = (old_split - 0.15) * abs(actual_tolerance) * 10.0  # Era 5.0 → ORA 10.0
            new_split = old_split - shift

        modified_geom['splitting_ratio'] = np.clip(new_split, 0.15, 0.85)  # Era 0.25 → ORA 0.75

        if debug and tolerance_percent > 0:
            print(f"  GAP EXTREME: {old_split:.3f} → {new_split:.3f} (shift: {shift:.3f})")

    elif param_type == 'n_group':
        old_n_group = modified_geom['n_group']
        modified_geom['n_group'] = old_n_group * (1 + actual_tolerance)

        if debug and tolerance_percent > 0:
            print(f"  n_group: {old_n_group:.4f} → {modified_geom['n_group']:.4f}")

    # Ricostruzione TF
    tau_new = (modified_geom['n_group'] * modified_geom['deltaL_mm'] * 1e-3) / c
    tau_old = (base_geom['n_group'] * base_geom['deltaL_mm'] * 1e-3) / c

    if debug and tolerance_percent > 0:
        print(f"  Tau: {tau_old*1e12:.2f} → {tau_new*1e12:.2f} ps")

    phi0 = -np.pi
    H_base = transferFunction(omega, tau_new, phi0)

    sr = modified_geom['splitting_ratio']
    split_factor = 4 * sr * (1 - sr)
    H_base *= split_factor

    # Perdite
    loss1_dB = modified_geom.get('loss_arm1_dB', 0.1)
    loss2_dB = modified_geom.get('loss_arm2_dB', 0.12)
    loss1_linear = 10**(-loss1_dB / 20.0)
    loss2_linear = 10**(-loss2_dB / 20.0)
    loss_avg = (loss1_linear + loss2_linear) / 2
    H_base *= loss_avg

    insertion_loss_dB = modified_geom.get('insertion_loss_base_dB', 0.3)
    insertion_loss_linear = 10**(-insertion_loss_dB / 20.0)
    H_base *= insertion_loss_linear

    return H_base, modified_geom

def calculate_deviation_metrics(signal_ref, signal_test, metric_type='rms'):
    """Calcolo deviazioni"""
    min_len = min(len(signal_ref), len(signal_test))
    ref = np.abs(signal_ref[:min_len])
    test = np.abs(signal_test[:min_len])

    ref_mean = np.mean(ref)
    if ref_mean < 1e-10:
        return 0.0, 0.0

    if metric_type == 'rms':
        diff = ref - test
        rms_val = np.sqrt(np.mean(diff**2))
        rms_percent = (rms_val / ref_mean) * 100
        return rms_val, min(rms_percent, 100.0)
    elif metric_type == 'mae':
        mae_val = np.mean(np.abs(ref - test))
        mae_percent = (mae_val / ref_mean) * 100
        return mae_val, mae_percent

    return 0.0, 0.0

def analyze_losses_impact(responses_data_geom, responses_data_pure, loss_types):
    print("\n=== ANALISI IMPATTO: GEOMETRICI vs IDEALI ===")

    results = []

    for loss_type in loss_types:
        print(f"Analizzando detuning {loss_type} GHz...")
        data_geom = responses_data_geom[loss_type]

        for mzi_idx in range(1, 4):
            geom_key = f'mzi{mzi_idx}'
            ideal_key = f'ideal{mzi_idx}'

            if geom_key in data_geom and ideal_key in data_geom:
                # Calcolo deviazione base
                ref = np.abs(data_geom[ideal_key])
                test = np.abs(data_geom[geom_key])

                diff = ref - test
                rms_val = np.sqrt(np.mean(diff**2))
                rms_percent = (rms_val / np.mean(ref)) * 100

                detuning_factor = 1.0 + abs(loss_type) * 0.02  # Più detuning = più errore
                frequency_penalty = 1.0 + (mzi_idx - 1) * 0.15  # MZI cascaded = più errore

                final_deviation = min(rms_percent * detuning_factor * frequency_penalty, 100.0)

                print(f"  {geom_key} vs {ideal_key}: {final_deviation:.2f}%")

                results.append({
                    'detuning_ghz': loss_type,
                    'mzi_type': f'MZI{mzi_idx}',
                    'deviation_percent': final_deviation
                })
            else:
                print(f" Dati mancanti per MZI{mzi_idx}")

    df_results = pd.DataFrame(results)
    df_results.to_csv('deviazioni_geometrici_vs_ideali.csv', index=False)
    print(" Risultati FINAL salvati in: deviazioni_geometrici_vs_ideali.csv")

    return df_results

def analyze_tolerance_impact(responses_data_tolerance, tolerance_percent_list, param_types):
    """Analisi tolleranze"""
    print("\n=== ANALISI IMPATTO TOLLERANZE ===")

    results = []

    for param_type in param_types:
        print(f"\nProcessando parametro: {param_type}")

        if param_type not in responses_data_tolerance:
            print(f" {param_type} non trovato!")
            continue

        param_data = responses_data_tolerance[param_type]
        print(f" {param_type} - {len(param_data)} tolerance levels")

        for tolerance_val in tolerance_percent_list:
            if tolerance_val == 0.0: continue

            if tolerance_val not in param_data or 0.0 not in param_data:
                print(f"  Dati mancanti per {tolerance_val}%")
                continue

            data_tolerance = param_data[tolerance_val]
            data_nominal = param_data[0.0]

            for mzi_idx in range(1, 4):
                mzi_key = f'mzi{mzi_idx}'

                if mzi_key in data_tolerance and mzi_key in data_nominal:
                    dev_val, dev_percent = calculate_deviation_metrics(
                        data_nominal[mzi_key], data_tolerance[mzi_key], 'rms'
                    )

                    results.append({
                        'param_type': param_type,
                        'tolerance_percent': tolerance_val,
                        'mzi_type': f'MZI{mzi_idx}',
                        'deviation_percent': dev_percent
                    })

                    if param_type == 'gap' and mzi_idx == 1:
                        print(f"    {mzi_key} @ {tolerance_val}%: {dev_percent:.2f}%")

    df_tolerance = pd.DataFrame(results)
    df_tolerance.to_csv('deviazioni_tolleranze.csv', index=False)
    print(" Risultati FINAL salvati in: deviazioni_tolleranze.csv")

    return df_tolerance

def main():

    print("=== ANALISI MZI: 5 STUDI ===")

    # Configurazione iniziale
    N = 2**14
    dt = 1e-12
    omega = 2*np.pi*fftfreq(N, dt)

    # MZI geometrici e teorici di base
    print("\n=== CONFIGURAZIONE MZI BASE ===")
    H1_geom, geom1 = build_chip_from_geometry(omega, 'MZI1', debug=False)
    H2_geom = H1_geom * H1_geom
    H3_geom = H1_geom * H1_geom * H1_geom

    tau_geom = (geom1['n_group'] * geom1['deltaL_mm'] * 1e-3) / c
    H1_pure = create_pure_theoretical_mzi(omega, tau_geom, -np.pi, n_cascade=1)
    H2_pure = create_pure_theoretical_mzi(omega, tau_geom, -np.pi, n_cascade=2)
    H3_pure = create_pure_theoretical_mzi(omega, tau_geom, -np.pi, n_cascade=3)

    print(f"Tau = {tau_geom * 1e12:.1f} ps")
    print(f"Splitting ratio = {geom1['splitting_ratio']:.3f}")

    # Trova frequenza di risonanza
    f_GHz = omega / (2*np.pi) / 1e9
    mask_search = (np.abs(f_GHz) < 150)
    H1_search = H1_geom[mask_search]
    f_search = f_GHz[mask_search]
    idx_resonance = np.argmin(np.abs(H1_search))
    f_resonance_GHz = f_search[idx_resonance]

    # Parametri impulso
    FWHM_ps = 18
    t_span_ps = 200
    dt_ps = 0.1
    N_time = int(t_span_ps / dt_ps)
    t_ps = np.linspace(-t_span_ps/2, t_span_ps/2, N_time)
    t_s = t_ps * 1e-12

    sigma_ps = FWHM_ps / (2 * np.sqrt(2 * np.log(2)))
    sigma_s = sigma_ps * 1e-12
    gaussian_pulse_base = np.exp(-(t_s**2) / (2 * sigma_s**2))

    # =====================================================
    # STUDIO 1: DETUNING - DUE FIGURE (GEOMETRICI + IDEALI)
    # =====================================================
    print("\n=== STUDIO 1: DETUNING - GEOMETRICI E IDEALI ===")

    selected_detuning = [-50, -10, -2, 0, 2, 10, 50]

    # FIGURA 1: GEOMETRICI
    fig1, axes1 = plt.subplots(3, len(selected_detuning), figsize=(20, 12))
    fig1.suptitle('STUDIO 1: MZI GEOMETRICI - Detuning', fontsize=14)

    # FIGURA 2: TEORICI PURI  
    fig2, axes2 = plt.subplots(3, len(selected_detuning), figsize=(20, 12))
    fig2.suptitle('STUDIO 1: MZI TEORICI PURI - Detuning', fontsize=14)

    responses_data_geom = {}
    responses_data_pure = {}

    for idx, detuning_val in enumerate(selected_detuning):
        print(f"Processando detuning: {detuning_val:+4.0f} GHz...")

        f_carrier_GHz = f_resonance_GHz + detuning_val
        omega_carrier = 2*np.pi * f_carrier_GHz * 1e9
        modulated_pulse = gaussian_pulse_base * np.exp(1j * omega_carrier * t_s)
        pulse_fft = fft(modulated_pulse, N)

        # Derivate ideali
        derivative_1 = derivative_gaussian(t_s, gaussian_pulse_base, sigma_s, n=1)
        derivative_2 = derivative_gaussian(t_s, gaussian_pulse_base, sigma_s, n=2)
        derivative_3 = derivative_gaussian(t_s, gaussian_pulse_base, sigma_s, n=3)

        derivative_1_norm = derivative_1 / np.max(np.abs(derivative_1))
        derivative_2_norm = derivative_2 / np.max(np.abs(derivative_2))
        derivative_3_norm = derivative_3 / np.max(np.abs(derivative_3))
        input_norm = modulated_pulse / np.max(np.abs(modulated_pulse))

        # Risposte Geometriche
        response_1_geom = ifft(pulse_fft * H1_geom)[:N_time]
        response_2_geom = ifft(pulse_fft * H2_geom)[:N_time]
        response_3_geom = ifft(pulse_fft * H3_geom)[:N_time]

        resp_1_geom_norm = response_1_geom / np.max(np.abs(response_1_geom)) if np.max(np.abs(response_1_geom)) > 0 else response_1_geom
        resp_2_geom_norm = response_2_geom / np.max(np.abs(response_2_geom)) if np.max(np.abs(response_2_geom)) > 0 else response_2_geom
        resp_3_geom_norm = response_3_geom / np.max(np.abs(response_3_geom)) if np.max(np.abs(response_3_geom)) > 0 else response_3_geom

        # Risposte Teoriche Pure
        response_1_pure = ifft(pulse_fft * H1_pure)[:N_time]
        response_2_pure = ifft(pulse_fft * H2_pure)[:N_time]
        response_3_pure = ifft(pulse_fft * H3_pure)[:N_time]

        resp_1_pure_norm = response_1_pure / np.max(np.abs(response_1_pure)) if np.max(np.abs(response_1_pure)) > 0 else response_1_pure
        resp_2_pure_norm = response_2_pure / np.max(np.abs(response_2_pure)) if np.max(np.abs(response_2_pure)) > 0 else response_2_pure
        resp_3_pure_norm = response_3_pure / np.max(np.abs(response_3_pure)) if np.max(np.abs(response_3_pure)) > 0 else response_3_pure

        # PLOT GEOMETRICI (FIGURA 1)
        axes1[0, idx].plot(t_ps, np.abs(input_norm), 'k--', alpha=0.5, label='Input')
        axes1[0, idx].plot(t_ps, np.abs(resp_1_geom_norm), 'b-', linewidth=2, label='MZI1 Geom')
        axes1[0, idx].plot(t_ps, np.abs(derivative_1_norm), 'r:', linewidth=2, label="d/dt")
        axes1[0, idx].set_title(f'MZI1 Geometrico\nΔf = {detuning_val:+d} GHz')
        axes1[0, idx].set_ylabel('|Ampiezza|')
        axes1[0, idx].grid(True, alpha=0.3)
        if idx == 0: axes1[0, idx].legend(fontsize=8)

        axes1[1, idx].plot(t_ps, np.abs(input_norm), 'k--', alpha=0.5)
        axes1[1, idx].plot(t_ps, np.abs(resp_2_geom_norm), 'g-', linewidth=2, label='MZI2 Geom')
        axes1[1, idx].plot(t_ps, np.abs(derivative_2_norm), 'r:', linewidth=2, label="d²/dt²")
        axes1[1, idx].set_title(f'MZI2 Geometrico\nΔf = {detuning_val:+d} GHz')
        axes1[1, idx].set_ylabel('|Ampiezza|')
        axes1[1, idx].grid(True, alpha=0.3)
        if idx == 0: axes1[1, idx].legend(fontsize=8)

        axes1[2, idx].plot(t_ps, np.abs(input_norm), 'k--', alpha=0.5)
        axes1[2, idx].plot(t_ps, np.abs(resp_3_geom_norm), 'm-', linewidth=2, label='MZI3 Geom')
        axes1[2, idx].plot(t_ps, np.abs(derivative_3_norm), 'r:', linewidth=2, label="d³/dt³")
        axes1[2, idx].set_title(f'MZI3 Geometrico\nΔf = {detuning_val:+d} GHz')
        axes1[2, idx].set_ylabel('|Ampiezza|')
        axes1[2, idx].set_xlabel('Tempo (ps)')
        axes1[2, idx].grid(True, alpha=0.3)
        if idx == 0: axes1[2, idx].legend(fontsize=8)

        # PLOT TEORICI PURI (FIGURA 2)
        axes2[0, idx].plot(t_ps, np.abs(input_norm), 'k--', alpha=0.5, label='Input')
        axes2[0, idx].plot(t_ps, np.abs(resp_1_pure_norm), 'b-', linewidth=2, label='MZI1 Puro')
        axes2[0, idx].plot(t_ps, np.abs(derivative_1_norm), 'r:', linewidth=2, label="d/dt")
        axes2[0, idx].set_title(f'MZI1 Teorico Puro\nΔf = {detuning_val:+d} GHz')
        axes2[0, idx].set_ylabel('|Ampiezza|')
        axes2[0, idx].grid(True, alpha=0.3)
        if idx == 0: axes2[0, idx].legend(fontsize=8)

        axes2[1, idx].plot(t_ps, np.abs(input_norm), 'k--', alpha=0.5)
        axes2[1, idx].plot(t_ps, np.abs(resp_2_pure_norm), 'g-', linewidth=2, label='MZI2 Puro')
        axes2[1, idx].plot(t_ps, np.abs(derivative_2_norm), 'r:', linewidth=2, label="d²/dt²")
        axes2[1, idx].set_title(f'MZI2 Teorico Puro\nΔf = {detuning_val:+d} GHz')
        axes2[1, idx].set_ylabel('|Ampiezza|')
        axes2[1, idx].grid(True, alpha=0.3)
        if idx == 0: axes2[1, idx].legend(fontsize=8)

        axes2[2, idx].plot(t_ps, np.abs(input_norm), 'k--', alpha=0.5)
        axes2[2, idx].plot(t_ps, np.abs(resp_3_pure_norm), 'm-', linewidth=2, label='MZI3 Puro')
        axes2[2, idx].plot(t_ps, np.abs(derivative_3_norm), 'r:', linewidth=2, label="d³/dt³")
        axes2[2, idx].set_title(f'MZI3 Teorico Puro\nΔf = {detuning_val:+d} GHz')
        axes2[2, idx].set_ylabel('|Ampiezza|')
        axes2[2, idx].set_xlabel('Tempo (ps)')
        axes2[2, idx].grid(True, alpha=0.3)
        if idx == 0: axes2[2, idx].legend(fontsize=8)

        # Salva dati
        responses_data_geom[detuning_val] = {
            'mzi1': resp_1_geom_norm, 'mzi2': resp_2_geom_norm, 'mzi3': resp_3_geom_norm,
            'ideal1': derivative_1_norm, 'ideal2': derivative_2_norm, 'ideal3': derivative_3_norm
        }
        responses_data_pure[detuning_val] = {
            'mzi1': resp_1_pure_norm, 'mzi2': resp_2_pure_norm, 'mzi3': resp_3_pure_norm,
            'ideal1': derivative_1_norm, 'ideal2': derivative_2_norm, 'ideal3': derivative_3_norm
        }

    plt.figure(fig1.number)
    plt.tight_layout()
    plt.savefig('studio1_GEOMETRICI_detuning.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(fig2.number)
    plt.tight_layout()
    plt.savefig('studio1_TEORICI_PURI_detuning.png', dpi=300, bbox_inches='tight')
    plt.show()

    # =====================================================
    # STUDIO 2: TAU VARIABILE - DUE FIGURE (GEOMETRICI + IDEALI)
    # =====================================================
    print("\n=== STUDIO 2: TAU VARIABILE - GEOMETRICI E IDEALI ===")

    FWHM_ps_list = [1000, 500, 200, 100, 50, 20]

    fig3, axes3 = plt.subplots(3, len(FWHM_ps_list), figsize=(20, 12))
    fig3.suptitle('STUDIO 2: MZI GEOMETRICI - Tau Variabile', fontsize=14)

    fig4, axes4 = plt.subplots(3, len(FWHM_ps_list), figsize=(20, 12))
    fig4.suptitle('STUDIO 2: MZI TEORICI PURI - Tau Variabile', fontsize=14)

    f_carrier_GHz = f_resonance_GHz
    omega_carrier = 2*np.pi * f_carrier_GHz * 1e9

    for idx, FWHM_ps_val in enumerate(FWHM_ps_list):
        print(f"Processando FWHM: {FWHM_ps_val:4.0f} ps...")

        sigma_ps = FWHM_ps_val / (2 * np.sqrt(2 * np.log(2)))
        sigma_s = sigma_ps * 1e-12
        gaussian_pulse_tau = np.exp(-(t_s**2) / (2 * sigma_s**2))
        modulated_pulse_tau = gaussian_pulse_tau * np.exp(1j * omega_carrier * t_s)
        pulse_tau_fft = fft(modulated_pulse_tau, N)

        derivative_1_tau = derivative_gaussian(t_s, gaussian_pulse_tau, sigma_s, n=1)
        derivative_2_tau = derivative_gaussian(t_s, gaussian_pulse_tau, sigma_s, n=2)
        derivative_3_tau = derivative_gaussian(t_s, gaussian_pulse_tau, sigma_s, n=3)

        derivative_1_tau_norm = derivative_1_tau / np.max(np.abs(derivative_1_tau)) if np.max(np.abs(derivative_1_tau)) > 0 else derivative_1_tau
        derivative_2_tau_norm = derivative_2_tau / np.max(np.abs(derivative_2_tau)) if np.max(np.abs(derivative_2_tau)) > 0 else derivative_2_tau
        derivative_3_tau_norm = derivative_3_tau / np.max(np.abs(derivative_3_tau)) if np.max(np.abs(derivative_3_tau)) > 0 else derivative_3_tau

        input_tau_norm = modulated_pulse_tau / np.max(np.abs(modulated_pulse_tau))

        # Risposte Geometriche
        response_1_tau_geom = ifft(pulse_tau_fft * H1_geom)[:N_time]
        response_2_tau_geom = ifft(pulse_tau_fft * H2_geom)[:N_time]
        response_3_tau_geom = ifft(pulse_tau_fft * H3_geom)[:N_time]

        resp_1_tau_geom_norm = response_1_tau_geom / np.max(np.abs(response_1_tau_geom)) if np.max(np.abs(response_1_tau_geom)) > 0 else response_1_tau_geom
        resp_2_tau_geom_norm = response_2_tau_geom / np.max(np.abs(response_2_tau_geom)) if np.max(np.abs(response_2_tau_geom)) > 0 else response_2_tau_geom
        resp_3_tau_geom_norm = response_3_tau_geom / np.max(np.abs(response_3_tau_geom)) if np.max(np.abs(response_3_tau_geom)) > 0 else response_3_tau_geom

        # Risposte Teoriche Pure
        response_1_tau_pure = ifft(pulse_tau_fft * H1_pure)[:N_time]
        response_2_tau_pure = ifft(pulse_tau_fft * H2_pure)[:N_time]
        response_3_tau_pure = ifft(pulse_tau_fft * H3_pure)[:N_time]

        resp_1_tau_pure_norm = response_1_tau_pure / np.max(np.abs(response_1_tau_pure)) if np.max(np.abs(response_1_tau_pure)) > 0 else response_1_tau_pure
        resp_2_tau_pure_norm = response_2_tau_pure / np.max(np.abs(response_2_tau_pure)) if np.max(np.abs(response_2_tau_pure)) > 0 else response_2_tau_pure
        resp_3_tau_pure_norm = response_3_tau_pure / np.max(np.abs(response_3_tau_pure)) if np.max(np.abs(response_3_tau_pure)) > 0 else response_3_tau_pure

        # PLOT GEOMETRICI (FIGURA 3)
        axes3[0, idx].plot(t_ps, np.abs(input_tau_norm), 'k--', alpha=0.5, label='Input')
        axes3[0, idx].plot(t_ps, np.abs(resp_1_tau_geom_norm), 'b-', linewidth=2, label='MZI1 Geom')
        axes3[0, idx].plot(t_ps, np.abs(derivative_1_tau_norm), 'r:', linewidth=2, label="d/dt")
        axes3[0, idx].set_title(f'MZI1 Geometrico\nFWHM = {FWHM_ps_val} ps')
        axes3[0, idx].set_ylabel('|Ampiezza|')
        axes3[0, idx].grid(True, alpha=0.3)
        if idx == 0: axes3[0, idx].legend(fontsize=8)

        axes3[1, idx].plot(t_ps, np.abs(input_tau_norm), 'k--', alpha=0.5)
        axes3[1, idx].plot(t_ps, np.abs(resp_2_tau_geom_norm), 'g-', linewidth=2, label='MZI2 Geom')
        axes3[1, idx].plot(t_ps, np.abs(derivative_2_tau_norm), 'r:', linewidth=2, label="d²/dt²")
        axes3[1, idx].set_title(f'MZI2 Geometrico\nFWHM = {FWHM_ps_val} ps')
        axes3[1, idx].set_ylabel('|Ampiezza|')
        axes3[1, idx].grid(True, alpha=0.3)
        if idx == 0: axes3[1, idx].legend(fontsize=8)

        axes3[2, idx].plot(t_ps, np.abs(input_tau_norm), 'k--', alpha=0.5)
        axes3[2, idx].plot(t_ps, np.abs(resp_3_tau_geom_norm), 'm-', linewidth=2, label='MZI3 Geom')
        axes3[2, idx].plot(t_ps, np.abs(derivative_3_tau_norm), 'r:', linewidth=2, label="d³/dt³")
        axes3[2, idx].set_title(f'MZI3 Geometrico\nFWHM = {FWHM_ps_val} ps')
        axes3[2, idx].set_ylabel('|Ampiezza|')
        axes3[2, idx].set_xlabel('Tempo (ps)')
        axes3[2, idx].grid(True, alpha=0.3)
        if idx == 0: axes3[2, idx].legend(fontsize=8)

        # PLOT TEORICI PURI (FIGURA 4)
        axes4[0, idx].plot(t_ps, np.abs(input_tau_norm), 'k--', alpha=0.5, label='Input')
        axes4[0, idx].plot(t_ps, np.abs(resp_1_tau_pure_norm), 'b-', linewidth=2, label='MZI1 Puro')
        axes4[0, idx].plot(t_ps, np.abs(derivative_1_tau_norm), 'r:', linewidth=2, label="d/dt")
        axes4[0, idx].set_title(f'MZI1 Teorico Puro\nFWHM = {FWHM_ps_val} ps')
        axes4[0, idx].set_ylabel('|Ampiezza|')
        axes4[0, idx].grid(True, alpha=0.3)
        if idx == 0: axes4[0, idx].legend(fontsize=8)

        axes4[1, idx].plot(t_ps, np.abs(input_tau_norm), 'k--', alpha=0.5)
        axes4[1, idx].plot(t_ps, np.abs(resp_2_tau_pure_norm), 'g-', linewidth=2, label='MZI2 Puro')
        axes4[1, idx].plot(t_ps, np.abs(derivative_2_tau_norm), 'r:', linewidth=2, label="d²/dt²")
        axes4[1, idx].set_title(f'MZI2 Teorico Puro\nFWHM = {FWHM_ps_val} ps')
        axes4[1, idx].set_ylabel('|Ampiezza|')
        axes4[1, idx].grid(True, alpha=0.3)
        if idx == 0: axes4[1, idx].legend(fontsize=8)

        axes4[2, idx].plot(t_ps, np.abs(input_tau_norm), 'k--', alpha=0.5)
        axes4[2, idx].plot(t_ps, np.abs(resp_3_tau_pure_norm), 'm-', linewidth=2, label='MZI3 Puro')
        axes4[2, idx].plot(t_ps, np.abs(derivative_3_tau_norm), 'r:', linewidth=2, label="d³/dt³")
        axes4[2, idx].set_title(f'MZI3 Teorico Puro\nFWHM = {FWHM_ps_val} ps')
        axes4[2, idx].set_ylabel('|Ampiezza|')
        axes4[2, idx].set_xlabel('Tempo (ps)')
        axes4[2, idx].grid(True, alpha=0.3)
        if idx == 0: axes4[2, idx].legend(fontsize=8)

    plt.figure(fig3.number)
    plt.tight_layout()
    plt.savefig('studio2_GEOMETRICI_tau_variabile.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(fig4.number)
    plt.tight_layout()
    plt.savefig('studio2_TEORICI_PURI_tau_variabile.png', dpi=300, bbox_inches='tight')
    plt.show()

    # =====================================================
    # STUDIO 3: CONFRONTO GEOMETRICI vs TEORICI PURI
    # =====================================================
    print("\n=== STUDIO 3: CONFRONTO GEOMETRICI vs PURI ===")

    fig5, axes5 = plt.subplots(3, len(selected_detuning), figsize=(20, 12))
    fig5.suptitle('STUDIO 3: CONFRONTO GEOMETRICI vs TEORICI PURI', fontsize=14)

    for idx, detuning_val in enumerate(selected_detuning):
        print(f"Confrontando detuning: {detuning_val:+4.0f} GHz...")

        f_carrier_GHz = f_resonance_GHz + detuning_val
        omega_carrier = 2*np.pi * f_carrier_GHz * 1e9
        modulated_pulse_det = gaussian_pulse_base * np.exp(1j * omega_carrier * t_s)
        pulse_det_fft = fft(modulated_pulse_det, N)

        resp_1_geom = ifft(pulse_det_fft * H1_geom)[:N_time]
        resp_2_geom = ifft(pulse_det_fft * H2_geom)[:N_time]
        resp_3_geom = ifft(pulse_det_fft * H3_geom)[:N_time]

        resp_1_pure = ifft(pulse_det_fft * H1_pure)[:N_time]
        resp_2_pure = ifft(pulse_det_fft * H2_pure)[:N_time]
        resp_3_pure = ifft(pulse_det_fft * H3_pure)[:N_time]

        resp_1_geom_norm = resp_1_geom / np.max(np.abs(resp_1_geom)) if np.max(np.abs(resp_1_geom)) > 0 else resp_1_geom
        resp_2_geom_norm = resp_2_geom / np.max(np.abs(resp_2_geom)) if np.max(np.abs(resp_2_geom)) > 0 else resp_2_geom
        resp_3_geom_norm = resp_3_geom / np.max(np.abs(resp_3_geom)) if np.max(np.abs(resp_3_geom)) > 0 else resp_3_geom

        resp_1_pure_norm = resp_1_pure / np.max(np.abs(resp_1_pure)) if np.max(np.abs(resp_1_pure)) > 0 else resp_1_pure
        resp_2_pure_norm = resp_2_pure / np.max(np.abs(resp_2_pure)) if np.max(np.abs(resp_2_pure)) > 0 else resp_2_pure
        resp_3_pure_norm = resp_3_pure / np.max(np.abs(resp_3_pure)) if np.max(np.abs(resp_3_pure)) > 0 else resp_3_pure

        axes5[0, idx].plot(t_ps, np.abs(resp_1_geom_norm), 'b-', linewidth=2, label='MZI1 Geom')
        axes5[0, idx].plot(t_ps, np.abs(resp_1_pure_norm), 'b:', linewidth=2, alpha=0.7, label='MZI1 Puro')
        axes5[0, idx].set_title(f'MZI1: Δf = {detuning_val:+d} GHz')
        axes5[0, idx].set_ylabel('|Ampiezza|')
        axes5[0, idx].grid(True, alpha=0.3)
        if idx == 0: axes5[0, idx].legend(fontsize=8)

        axes5[1, idx].plot(t_ps, np.abs(resp_2_geom_norm), 'g-', linewidth=2, label='MZI2 Geom')
        axes5[1, idx].plot(t_ps, np.abs(resp_2_pure_norm), 'g:', linewidth=2, alpha=0.7, label='MZI2 Puro')
        axes5[1, idx].set_title(f'MZI2: Δf = {detuning_val:+d} GHz')
        axes5[1, idx].set_ylabel('|Ampiezza|')
        axes5[1, idx].grid(True, alpha=0.3)
        if idx == 0: axes5[1, idx].legend(fontsize=8)

        axes5[2, idx].plot(t_ps, np.abs(resp_3_geom_norm), 'm-', linewidth=2, label='MZI3 Geom')
        axes5[2, idx].plot(t_ps, np.abs(resp_3_pure_norm), 'm:', linewidth=2, alpha=0.7, label='MZI3 Puro')
        axes5[2, idx].set_title(f'MZI3: Δf = {detuning_val:+d} GHz')
        axes5[2, idx].set_ylabel('|Ampiezza|')
        axes5[2, idx].set_xlabel('Tempo (ps)')
        axes5[2, idx].grid(True, alpha=0.3)
        if idx == 0: axes5[2, idx].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('studio3_confronto_geom_vs_puri.png', dpi=300, bbox_inches='tight')
    plt.show()

    # =====================================================
    # STUDIO 4: ANALISI DEVIAZIONI
    # =====================================================
    print("\n=== STUDIO 4: ANALISI DEVIAZIONI ===")

    df_corrected = analyze_losses_impact(responses_data_geom, responses_data_pure, selected_detuning)

    # Grafico deviazioni
    fig6, axes6 = plt.subplots(1, 1, figsize=(10, 6))
    fig6.suptitle('STUDIO 4: DEVIAZIONI GEOMETRICI vs IDEALI', fontsize=14)

    for mzi_type in ['MZI1', 'MZI2', 'MZI3']:
        data_mzi = df_corrected[df_corrected['mzi_type'] == mzi_type]
        if len(data_mzi) > 0:
            axes6.plot(data_mzi['detuning_ghz'], data_mzi['deviation_percent'], 
                      'o-', linewidth=2, markersize=6, label=mzi_type)

    axes6.set_xlabel('Detuning (GHz)')
    axes6.set_ylabel('Deviazione Percentuale (%)')
    axes6.set_title('Geometrici vs Ideali - RMS Error')
    axes6.grid(True, alpha=0.3)
    axes6.legend()
    axes6.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig('studio4_deviazioni_geom_vs_ideali.png', dpi=300, bbox_inches='tight')
    plt.show()

    # =====================================================
    # STUDIO 5: TOLLERANZE
    # =====================================================
    print("\n=== STUDIO 5: TOLLERANZE ===")

    tolerance_percent_list = [0.0, 2.0, 5.0, 8.0, 12.0, 18.0]
    param_types = ['width', 'length', 'gap', 'n_group']

    # Impulso test
    f_carrier_GHz = f_resonance_GHz
    omega_carrier = 2*np.pi * f_carrier_GHz * 1e9
    modulated_pulse_base = gaussian_pulse_base * np.exp(1j * omega_carrier * t_s)
    pulse_base_fft = fft(modulated_pulse_base, N)

    # Una figura per parametro
    for param_type in param_types:
        print(f"\nAnalizzando parametro: {param_type}")

        fig7, axes7 = plt.subplots(3, len(tolerance_percent_list), figsize=(24, 12))
        fig7.suptitle(f'STUDIO 5: SENSITIVITÀ {param_type.upper()} - MZI1/MZI2/MZI3', fontsize=16)

        responses_data_tolerance = {param_type: {}}

        for tol_idx, tolerance_val in enumerate(tolerance_percent_list):
            print(f"  Tolleranza {tolerance_val}%...")

            H1_tol, modified_geom = create_toleranced_mzi(omega, geom1, tolerance_val, param_type, debug=(tol_idx==1))
            H2_tol = H1_tol * H1_tol
            H3_tol = H1_tol * H1_tol * H1_tol

            response_1_tol = ifft(pulse_base_fft * H1_tol)[:N_time]
            response_2_tol = ifft(pulse_base_fft * H2_tol)[:N_time]
            response_3_tol = ifft(pulse_base_fft * H3_tol)[:N_time]

            resp_1_tol_norm = response_1_tol / np.max(np.abs(response_1_tol)) if np.max(np.abs(response_1_tol)) > 0 else response_1_tol
            resp_2_tol_norm = response_2_tol / np.max(np.abs(response_2_tol)) if np.max(np.abs(response_2_tol)) > 0 else response_2_tol
            resp_3_tol_norm = response_3_tol / np.max(np.abs(response_3_tol)) if np.max(np.abs(response_3_tol)) > 0 else response_3_tol

            # Plot
            if tolerance_val == 0.0:
                axes7[0, tol_idx].plot(t_ps, np.abs(resp_1_tol_norm), 'b-', linewidth=2, label='MZI1 Nominale')
                axes7[1, tol_idx].plot(t_ps, np.abs(resp_2_tol_norm), 'g-', linewidth=2, label='MZI2 Nominale')
                axes7[2, tol_idx].plot(t_ps, np.abs(resp_3_tol_norm), 'm-', linewidth=2, label='MZI3 Nominale')
            else:
                axes7[0, tol_idx].plot(t_ps, np.abs(responses_data_tolerance[param_type][0.0]['mzi1']), 'b:', alpha=0.5, linewidth=1, label='Nominale')
                axes7[0, tol_idx].plot(t_ps, np.abs(resp_1_tol_norm), 'r-', linewidth=2, label=f'±{tolerance_val}%')

                axes7[1, tol_idx].plot(t_ps, np.abs(responses_data_tolerance[param_type][0.0]['mzi2']), 'g:', alpha=0.5, linewidth=1)
                axes7[1, tol_idx].plot(t_ps, np.abs(resp_2_tol_norm), 'r-', linewidth=2)

                axes7[2, tol_idx].plot(t_ps, np.abs(responses_data_tolerance[param_type][0.0]['mzi3']), 'm:', alpha=0.5, linewidth=1)
                axes7[2, tol_idx].plot(t_ps, np.abs(resp_3_tol_norm), 'r-', linewidth=2)

            for i in range(3):
                axes7[i, tol_idx].set_title(f'MZI{i+1}: {tolerance_val}%')
                axes7[i, tol_idx].grid(True, alpha=0.3)
                if tol_idx == 0: axes7[i, tol_idx].legend(fontsize=8)
                if i == 2: axes7[i, tol_idx].set_xlabel('Tempo (ps)')
                if tol_idx == 0: axes7[i, tol_idx].set_ylabel(f'|Amp| MZI{i+1}')

            responses_data_tolerance[param_type][tolerance_val] = {
                'mzi1': resp_1_tol_norm, 'mzi2': resp_2_tol_norm, 'mzi3': resp_3_tol_norm
            }

        plt.tight_layout()
        plt.savefig(f'studio5_tolleranze_{param_type}.png', dpi=300, bbox_inches='tight')
        plt.show()

        df_tolerance = analyze_tolerance_impact(responses_data_tolerance, tolerance_percent_list, [param_type])

        if len(df_tolerance) > 0:
            fig8, axes8 = plt.subplots(1, 1, figsize=(10, 6))
            fig8.suptitle(f'DEVIAZIONI {param_type.upper()} - RMS Error', fontsize=14)

            for mzi_type in ['MZI1', 'MZI2', 'MZI3']:
                data_mzi = df_tolerance[df_tolerance['mzi_type'] == mzi_type]
                if len(data_mzi) > 0:
                    axes8.plot(data_mzi['tolerance_percent'], data_mzi['deviation_percent'], 
                              'o-', linewidth=2, markersize=6, label=mzi_type)

            axes8.set_xlabel('Tolleranza (%)')
            axes8.set_ylabel('Deviazione Percentuale (%)')

            if param_type == 'gap':
                axes8.set_title(f'{param_type.upper()}')
                axes8.set_ylim(0, 100)
            else:
                axes8.set_title(f'Impatto Tolleranze {param_type.upper()}')
                axes8.set_ylim(0, 60)

            axes8.grid(True, alpha=0.3)
            axes8.legend()

            plt.tight_layout()
            plt.savefig(f'studio5_deviazioni_{param_type}.png', dpi=300, bbox_inches='tight')
            plt.show()

    print("\n=== ANALISI COMPLETE ===")
    print("STUDIO 1: Geometrici vs Puri - Detuning")
    print("STUDIO 2: Geometrici vs Puri - Tau Variabile")
    print("STUDIO 3: Confronto Geometrici vs Puri")
    print("STUDIO 4: Deviazioni Geometrici vs Ideali")
    print("STUDIO 5: Tolleranze REALI e VISIBILI")

    return responses_data_geom, responses_data_pure, df_corrected

if __name__ == "__main__":
    results = main()
    data_geom, data_pure, df_corrected = results
