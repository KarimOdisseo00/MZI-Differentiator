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

# Importa il modulo Prova.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from MZI import *

def create_pure_theoretical_mzi(omega, tau, phi0=-np.pi, n_cascade=1):
    H_single = transferFunction(omega, tau, phi0)
    H_pure = H_single ** n_cascade
    return H_pure

def create_toleranced_mzi(omega, base_geom, tolerance_percent, param_type, debug=True):
    """
        Args:
        omega: Array frequenze angolari
        base_geom: Parametri geometrici di riferimento da build_chip_from_geometry
        tolerance_percent: Tolleranza in percentuale
        param_type: Tipo parametro ('width', 'length', 'gap', 'n_group')
        debug: Stampa debug dei cambiamenti

    Returns:
        H_modified: Transfer function modificata 
        modified_geom: Parametri geometrici modificati
    """
    if debug:
        print(f"\n--- TOLLERANZA {tolerance_percent}% su {param_type.upper()} ---")

    # COPIA i parametri geometrici
    modified_geom = base_geom.copy()

    # Amplifica le tolleranze per renderle visibili
    amplification = 3.0  # Fattore di amplificazione per visibilità
    actual_tolerance = tolerance_percent * amplification / 100.0

    if param_type == 'width':
        # Larghezza guide n_eff e n_group cambiano drasticamente
        old_n_eff = modified_geom.get('n_eff', 2.4)
        old_n_group = modified_geom['n_group']

        modified_geom['n_eff'] = old_n_eff * (1 + actual_tolerance)
        modified_geom['n_group'] = old_n_group * (1 + actual_tolerance)

        if debug:
            print(f"  n_eff:   {old_n_eff:.4f} -> {modified_geom['n_eff']:.4f}")
            print(f"  n_group: {old_n_group:.4f} -> {modified_geom['n_group']:.4f}")

    elif param_type == 'length':
        old_deltaL = modified_geom['deltaL_mm']
        modified_geom['deltaL_mm'] = old_deltaL * (1 + actual_tolerance)

        if debug:
            print(f"  deltaL_mm: {old_deltaL:.4f} -> {modified_geom['deltaL_mm']:.4f}")

    elif param_type == 'gap':
        old_split = modified_geom['splitting_ratio']

        if actual_tolerance > 0:
            shift = (0.5 - old_split) * abs(actual_tolerance) * 2.5  
            new_split = old_split + shift
        else:
            shift = (old_split - 0.3) * abs(actual_tolerance) * 2.5
            new_split = old_split - shift

        modified_geom['splitting_ratio'] = np.clip(new_split, 0.25, 0.75)  # Range fisico

        if debug:
            print(f"  splitting_ratio: {old_split:.4f} → {modified_geom['splitting_ratio']:.4f}")

    elif param_type == 'n_group':
        old_n_group = modified_geom['n_group']
        modified_geom['n_group'] = old_n_group * (1 + actual_tolerance)

        if debug:
            print(f"  n_group: {old_n_group:.4f} -> {modified_geom['n_group']:.4f}")

    tau_new = (modified_geom['n_group'] * modified_geom['deltaL_mm'] * 1e-3) / c
    tau_old = (base_geom['n_group'] * base_geom['deltaL_mm'] * 1e-3) / c

    if debug:
        print(f"  Tau: {tau_old*1e12:.2f} → {tau_new*1e12:.2f} ps")

    phi0 = -np.pi  # Fase fissa

    H_base = transferFunction(omega, tau_new, phi0)

    # Applica il nuovo splitting ratio 
    sr = modified_geom['splitting_ratio']
    split_factor = 4 * sr * (1 - sr)  # Formula corretta per MZI
    H_base *= split_factor

    # Applica perdite per braccio 
    loss1_dB = modified_geom.get('loss_arm1_dB', 0.1)
    loss2_dB = modified_geom.get('loss_arm2_dB', 0.12)
    loss1_linear = 10**(-loss1_dB / 20.0)
    loss2_linear = 10**(-loss2_dB / 20.0)
    loss_avg = (loss1_linear + loss2_linear) / 2
    H_base *= loss_avg

    # Applica insertion loss base
    insertion_loss_dB = modified_geom.get('insertion_loss_base_dB', 0.3)
    insertion_loss_linear = 10**(-insertion_loss_dB / 20.0)
    H_base *= insertion_loss_linear

    # Profondità notch
    if debug:
        info_new = notch_info(H_base, omega)
        info_old = notch_info(transferFunction(omega, tau_old, phi0), omega)
        print(f"  Notch depth: {info_old['depth_db']:.1f} → {info_new['depth_db']:.1f} dB")

    return H_base, modified_geom

def calculate_deviation_metrics(signal_ref, signal_test, metric_type='rms'):
    """Calcola metriche di deviazione tra segnale di riferimento e test"""
    min_len = min(len(signal_ref), len(signal_test))
    ref = np.abs(signal_ref[:min_len])
    test = np.abs(signal_test[:min_len])

    if metric_type == 'rms':
        deviation = np.sqrt(np.mean((ref - test)**2))
        deviation_percent = (deviation / np.mean(ref)) * 100 if np.mean(ref) > 0 else 0
    elif metric_type == 'chi2':
        epsilon = 1e-10
        chi2 = np.sum((ref - test)**2 / (ref + epsilon))
        deviation = chi2 / len(ref)
        deviation_percent = (deviation / np.mean(ref)) * 100 if np.mean(ref) > 0 else 0
    elif metric_type == 'mae':
        deviation = np.mean(np.abs(ref - test))
        deviation_percent = (deviation / np.mean(ref)) * 100 if np.mean(ref) > 0 else 0
    elif metric_type == 'max_error':
        deviation = np.max(np.abs(ref - test))
        deviation_percent = (deviation / np.max(ref)) * 100 if np.max(ref) > 0 else 0

    return deviation, deviation_percent

def analyze_losses_impact_and_save_corrected(responses_data_geom, responses_data_pure, loss_types):
    """Analizza Geometrici vs Ideali"""
    print("\n=== ANALISI IMPATTO: GEOMETRICI vs IDEALI ===")

    results = []
    metrics = ['rms', 'chi2', 'mae', 'max_error']

    for loss_type in loss_types:
        print(f"\nAnalizzando detuning {loss_type}...")
        data_geom = responses_data_geom[loss_type]
        data_pure = responses_data_pure[loss_type]

        # Geometrici vs Ideali 
        for mzi_idx in range(1, 4):
            geom_key = f'mzi{mzi_idx}'
            ideal_key = f'ideal{mzi_idx}'

            if geom_key in data_geom and ideal_key in data_geom:
                for metric in metrics:
                    dev_val, dev_percent = calculate_deviation_metrics(
                        data_geom[ideal_key], data_geom[geom_key], metric
                    )

                    results.append({
                        'detuning_ghz': loss_type,
                        'mzi_type': f'MZI{mzi_idx}',
                        'metric': metric,
                        'deviation_value': dev_val,
                        'deviation_percent': dev_percent,
                        'comparison': 'Ideale_vs_Geometrico'
                    })

    df_results = pd.DataFrame(results)
    df_results.to_csv('analisi_deviazioni_geometrici_vs_ideali.csv', index=False)
    print("\n Risultati salvati in: analisi_deviazioni_geometrici_vs_ideali.csv")

    summary_stats = df_results.groupby(['detuning_ghz', 'mzi_type', 'metric']).agg({
        'deviation_percent': ['mean', 'std', 'min', 'max']
    }).round(3)

    summary_stats.to_csv('summary_deviazioni_geometrici_vs_ideali.csv')
    print(" Summary salvato in: summary_deviazioni_geometrici_vs_ideali.csv")

    return df_results

def analyze_tolerance_impact_and_save(responses_data_tolerance, tolerance_percent_list, param_types):
    """Analizza l'impatto delle tolleranze e salva i risultati"""
    print("\n=== ANALISI IMPATTO TOLLERANZE ===")

    results_tolerance = []
    metrics = ['rms', 'chi2', 'mae', 'max_error']

    for param_type in param_types:
        for tolerance_val in tolerance_percent_list:
            if tolerance_val == 0.0: continue

            print(f"Analizzando {param_type} con tolleranza {tolerance_val}%...")

            data_tolerance = responses_data_tolerance[param_type][tolerance_val]
            data_nominal = responses_data_tolerance[param_type][0.0]

            for mzi_idx in range(1, 4):
                mzi_key = f'mzi{mzi_idx}'

                if mzi_key in data_tolerance and mzi_key in data_nominal:
                    for metric in metrics:
                        dev_val, dev_percent = calculate_deviation_metrics(
                            data_nominal[mzi_key], data_tolerance[mzi_key], metric
                        )

                        results_tolerance.append({
                            'param_type': param_type,
                            'tolerance_percent': tolerance_val,
                            'mzi_type': f'MZI{mzi_idx}',
                            'metric': metric,
                            'deviation_value': dev_val,
                            'deviation_percent': dev_percent
                        })

    df_tolerance = pd.DataFrame(results_tolerance)
    df_tolerance.to_csv('analisi_deviazioni_tolleranze.csv', index=False)
    print("\n Risultati salvati in: analisi_deviazioni_tolleranze.csv")

    summary_stats = df_tolerance.groupby(['param_type', 'tolerance_percent', 'mzi_type', 'metric']).agg({
        'deviation_percent': ['mean', 'std', 'min', 'max']
    }).round(3)

    summary_stats.to_csv('summary_deviazioni_tolleranze.csv')
    print(" Summary salvato in: summary_deviazioni_tolleranze.csv")

    return df_tolerance

def main():
    """Funzione principale"""
    print("=== ANALISI MZI COMPLETA ===")

    # Configurazione iniziale
    N = 2**14
    dt = 1e-12
    omega = 2*np.pi*fftfreq(N, dt)

    # =====================================================
    # MZI GEOMETRICI E TEORICI PURI DI BASE
    # =====================================================
    print("\n=== CONFIGURAZIONE MZI BASE ===")
    H1_geom, geom1 = build_chip_from_geometry(omega, 'MZI1', debug=False)
    H2_geom = H1_geom * H1_geom
    H3_geom = H1_geom * H1_geom * H1_geom

    tau_geom = (geom1['n_group'] * geom1['deltaL_mm'] * 1e-3) / c
    tau_pure = tau_geom
    phi0_pure = -np.pi

    H1_pure = create_pure_theoretical_mzi(omega, tau_pure, phi0_pure, n_cascade=1)
    H2_pure = create_pure_theoretical_mzi(omega, tau_pure, phi0_pure, n_cascade=2)
    H3_pure = create_pure_theoretical_mzi(omega, tau_pure, phi0_pure, n_cascade=3)

    print(f"Tau = {tau_geom * 1e12:.1f} ps")
    print(f"Splitting ratio = {geom1['splitting_ratio']:.3f}")

    # Mostra parametri geometrici di base
    print("\nParametri geometrici di riferimento:")
    for key, value in geom1.items():
        print(f"  {key}: {value}")

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
    # STUDIO 1: GEOMETRICI vs TEORICI PURI - DETUNING
    # =====================================================
    print("\n=== STUDIO 1: GEOMETRICI vs TEORICI PURI - DETUNING ===")

    selected_detuning = [-50, -10, -2, 0, 2, 10, 50]

    fig1, axes1 = plt.subplots(3, len(selected_detuning), figsize=(20, 12))
    fig1.suptitle('STUDIO 1: MZI GEOMETRICI - Detuning', fontsize=14)

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

        # Plot Geometrici
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

        # Risposte Teoriche Pure
        response_1_pure = ifft(pulse_fft * H1_pure)[:N_time]
        response_2_pure = ifft(pulse_fft * H2_pure)[:N_time]
        response_3_pure = ifft(pulse_fft * H3_pure)[:N_time]

        resp_1_pure_norm = response_1_pure / np.max(np.abs(response_1_pure)) if np.max(np.abs(response_1_pure)) > 0 else response_1_pure
        resp_2_pure_norm = response_2_pure / np.max(np.abs(response_2_pure)) if np.max(np.abs(response_2_pure)) > 0 else response_2_pure
        resp_3_pure_norm = response_3_pure / np.max(np.abs(response_3_pure)) if np.max(np.abs(response_3_pure)) > 0 else response_3_pure

        # Plot Teorici Puri
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
    plt.savefig('studio1_geometrici_detuning.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(fig2.number)
    plt.tight_layout()
    plt.savefig('studio1_teorici_puri_detuning.png', dpi=300, bbox_inches='tight')
    plt.show()

    # =====================================================
    # STUDIO 2: GEOMETRICI vs TEORICI PURI - TAU VARIABILE
    # =====================================================
    print("\n=== STUDIO 2: GEOMETRICI vs TEORICI PURI - TAU VARIABILE ===")

    FWHM_ps_list = [1000, 500, 200, 100, 50, 20]

    fig3, axes3 = plt.subplots(3, len(FWHM_ps_list), figsize=(20, 12))
    fig3.suptitle('STUDIO 2: MZI GEOMETRICI - Tau Variabile', fontsize=14)

    fig4, axes4 = plt.subplots(3, len(FWHM_ps_list), figsize=(20, 12))
    fig4.suptitle('STUDIO 2: MZI TEORICI PURI - Tau Variabile', fontsize=14)

    responses_data_tau_geom = {}
    responses_data_tau_pure = {}
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

        # Plot Geometrici
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

        # Risposte Teoriche Pure
        response_1_tau_pure = ifft(pulse_tau_fft * H1_pure)[:N_time]
        response_2_tau_pure = ifft(pulse_tau_fft * H2_pure)[:N_time]
        response_3_tau_pure = ifft(pulse_tau_fft * H3_pure)[:N_time]

        resp_1_tau_pure_norm = response_1_tau_pure / np.max(np.abs(response_1_tau_pure)) if np.max(np.abs(response_1_tau_pure)) > 0 else response_1_tau_pure
        resp_2_tau_pure_norm = response_2_tau_pure / np.max(np.abs(response_2_tau_pure)) if np.max(np.abs(response_2_tau_pure)) > 0 else response_2_tau_pure
        resp_3_tau_pure_norm = response_3_tau_pure / np.max(np.abs(response_3_tau_pure)) if np.max(np.abs(response_3_tau_pure)) > 0 else response_3_tau_pure

        # Plot Teorici Puri
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

        # Salva dati
        responses_data_tau_geom[FWHM_ps_val] = {
            'mzi1': resp_1_tau_geom_norm, 'mzi2': resp_2_tau_geom_norm, 'mzi3': resp_3_tau_geom_norm,
            'ideal1': derivative_1_tau_norm, 'ideal2': derivative_2_tau_norm, 'ideal3': derivative_3_tau_norm
        }

        responses_data_tau_pure[FWHM_ps_val] = {
            'mzi1': resp_1_tau_pure_norm, 'mzi2': resp_2_tau_pure_norm, 'mzi3': resp_3_tau_pure_norm,
            'ideal1': derivative_1_tau_norm, 'ideal2': derivative_2_tau_norm, 'ideal3': derivative_3_tau_norm
        }

    plt.figure(fig3.number)
    plt.tight_layout()
    plt.savefig('studio2_geometrici_tau_variabile.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(fig4.number)
    plt.tight_layout()
    plt.savefig('studio2_teorici_puri_tau_variabile.png', dpi=300, bbox_inches='tight')
    plt.show()

    # =====================================================
    # STUDIO 3: GEOMETRICI vs TEORICI PURI
    # =====================================================
    print("\n=== STUDIO 3: GEOMETRICI vs TEORICI PURI ===")

    fig5, axes5 = plt.subplots(3, len(selected_detuning), figsize=(20, 12))
    fig5.suptitle('STUDIO 3: CONFRONTO GEOMETRICI vs TEORICI PURI', fontsize=14)

    modulated_pulse_base = gaussian_pulse_base * np.exp(1j * omega_carrier * t_s)

    derivative_1 = derivative_gaussian(t_s, gaussian_pulse_base, sigma_s, n=1)
    derivative_2 = derivative_gaussian(t_s, gaussian_pulse_base, sigma_s, n=2)
    derivative_3 = derivative_gaussian(t_s, gaussian_pulse_base, sigma_s, n=3)

    derivative_1_norm = derivative_1 / np.max(np.abs(derivative_1))
    derivative_2_norm = derivative_2 / np.max(np.abs(derivative_2))
    derivative_3_norm = derivative_3 / np.max(np.abs(derivative_3))

    for idx, detuning_val in enumerate(selected_detuning):
        print(f"Confrontando a detuning: {detuning_val:+4.0f} GHz...")

        f_carrier_GHz = f_resonance_GHz + detuning_val
        omega_carrier = 2*np.pi * f_carrier_GHz * 1e9
        modulated_pulse_det = gaussian_pulse_base * np.exp(1j * omega_carrier * t_s)
        pulse_det_fft = fft(modulated_pulse_det, N)

        # Risposte a questo detuning
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

        # Plot confronto diretto
        axes5[0, idx].plot(t_ps, np.abs(resp_1_geom_norm), 'b-', linewidth=2, label='MZI1 Geom')
        axes5[0, idx].plot(t_ps, np.abs(resp_1_pure_norm), 'b:', linewidth=2, alpha=0.7, label='MZI1 Puro')
        axes5[0, idx].plot(t_ps, np.abs(derivative_1_norm), 'r:', linewidth=1, alpha=0.5, label="d/dt")
        axes5[0, idx].set_title(f'MZI1: Δf = {detuning_val:+d} GHz')
        axes5[0, idx].set_ylabel('|Ampiezza|')
        axes5[0, idx].grid(True, alpha=0.3)
        if idx == 0: axes5[0, idx].legend(fontsize=8)

        axes5[1, idx].plot(t_ps, np.abs(resp_2_geom_norm), 'g-', linewidth=2, label='MZI2 Geom')
        axes5[1, idx].plot(t_ps, np.abs(resp_2_pure_norm), 'g:', linewidth=2, alpha=0.7, label='MZI2 Puro')
        axes5[1, idx].plot(t_ps, np.abs(derivative_2_norm), 'r:', linewidth=1, alpha=0.5, label="d²/dt²")
        axes5[1, idx].set_title(f'MZI2: Δf = {detuning_val:+d} GHz')
        axes5[1, idx].set_ylabel('|Ampiezza|')
        axes5[1, idx].grid(True, alpha=0.3)
        if idx == 0: axes5[1, idx].legend(fontsize=8)

        axes5[2, idx].plot(t_ps, np.abs(resp_3_geom_norm), 'm-', linewidth=2, label='MZI3 Geom')
        axes5[2, idx].plot(t_ps, np.abs(resp_3_pure_norm), 'm:', linewidth=2, alpha=0.7, label='MZI3 Puro')
        axes5[2, idx].plot(t_ps, np.abs(derivative_3_norm), 'r:', linewidth=1, alpha=0.5, label="d³/dt³")
        axes5[2, idx].set_title(f'MZI3: Δf = {detuning_val:+d} GHz')
        axes5[2, idx].set_ylabel('|Ampiezza|')
        axes5[2, idx].set_xlabel('Tempo (ps)')
        axes5[2, idx].grid(True, alpha=0.3)
        if idx == 0: axes5[2, idx].legend(fontsize=8)

    plt.figure(fig5.number)
    plt.tight_layout()
    plt.savefig('studio3_confronto_geometrici_vs_puri.png', dpi=300, bbox_inches='tight')
    plt.show()

    # =====================================================
    # STUDIO 4: ANALISI DEVIAZIONI GEOMETRICI vs IDEALI
    # =====================================================
    df_corrected = analyze_losses_impact_and_save_corrected(
        responses_data_geom, responses_data_pure, selected_detuning
    )

    # Grafico deviazioni corrette
    df_rms = df_corrected[df_corrected['metric'] == 'rms'].copy()

    fig6, axes6 = plt.subplots(1, 1, figsize=(10, 6))
    fig6.suptitle('STUDIO 4: DEVIAZIONI GEOMETRICI vs IDEALI', fontsize=14)

    for mzi_type in ['MZI1', 'MZI2', 'MZI3']:
        data_mzi = df_rms[df_rms['mzi_type'] == mzi_type]
        if len(data_mzi) > 0:
            axes6.plot(data_mzi['detuning_ghz'], data_mzi['deviation_percent'], 
                      'o-', linewidth=2, markersize=6, label=mzi_type)

    axes6.set_xlabel('Detuning (GHz)')
    axes6.set_ylabel('Deviazione Percentuale (%)')
    axes6.set_title('Geometrici vs Ideali - RMS Error')
    axes6.grid(True, alpha=0.3)
    axes6.legend()
    axes6.set_yscale('log')

    plt.tight_layout()
    plt.savefig('studio4_deviazioni_geometrici_vs_ideali.png', dpi=300, bbox_inches='tight')
    plt.show()

    # =====================================================
    # STUDIO 5: TOLLERANZE GEOMETRICHE CORRETTE
    # =====================================================
    print("\n=== STUDIO 5: TOLLERANZE GEOMETRICHE CORRETTE ===")

    # Tolleranze AMPLIFICATE per essere MOLTO visibili
    tolerance_percent_list = [0.0, 2.0, 5.0, 8.0, 12.0, 18.0]
    param_types = ['width', 'length', 'gap', 'n_group']

    # Impulso per test tolleranze (alla risonanza)
    pulse_base_fft = fft(modulated_pulse_base, N)

    # UNA FIGURA per PARAMETRO per chiarezza
    for param_type in param_types:
        print(f"\nAnalizzando parametro: {param_type}")

        fig7, axes7 = plt.subplots(3, len(tolerance_percent_list), figsize=(24, 12))
        fig7.suptitle(f'STUDIO 5: SENSITIVITÀ {param_type.upper()} - MZI1/MZI2/MZI3', fontsize=16)

        responses_data_tolerance = {param_type: {}}

        for tol_idx, tolerance_val in enumerate(tolerance_percent_list):
            print(f"  Tolleranza {tolerance_val}%...")

            # Modifica parametri REALI 
            H1_tol, modified_geom = create_toleranced_mzi(omega, geom1, tolerance_val, param_type, debug=(tol_idx==1))
            H2_tol = H1_tol * H1_tol
            H3_tol = H1_tol * H1_tol * H1_tol

            # Calcola risposte per TUTTI e 3 gli MZI
            response_1_tol = ifft(pulse_base_fft * H1_tol)[:N_time]
            response_2_tol = ifft(pulse_base_fft * H2_tol)[:N_time]
            response_3_tol = ifft(pulse_base_fft * H3_tol)[:N_time]

            resp_1_tol_norm = response_1_tol / np.max(np.abs(response_1_tol)) if np.max(np.abs(response_1_tol)) > 0 else response_1_tol
            resp_2_tol_norm = response_2_tol / np.max(np.abs(response_2_tol)) if np.max(np.abs(response_2_tol)) > 0 else response_2_tol
            resp_3_tol_norm = response_3_tol / np.max(np.abs(response_3_tol)) if np.max(np.abs(response_3_tol)) > 0 else response_3_tol

            # Plot MZI1
            if tolerance_val == 0.0:
                axes7[0, tol_idx].plot(t_ps, np.abs(resp_1_tol_norm), 'b-', linewidth=2, label='MZI1 Nominale')
            else:
                axes7[0, tol_idx].plot(t_ps, np.abs(responses_data_tolerance[param_type][0.0]['mzi1']), 'b:', alpha=0.5, linewidth=1, label='Nominale')
                axes7[0, tol_idx].plot(t_ps, np.abs(resp_1_tol_norm), 'r-', linewidth=2, label=f'±{tolerance_val}%')

            axes7[0, tol_idx].set_title(f'MZI1: {tolerance_val}%', fontsize=12)
            axes7[0, tol_idx].set_ylabel('|Ampiezza| MZI1')
            axes7[0, tol_idx].grid(True, alpha=0.3)
            if tol_idx == 0: axes7[0, tol_idx].legend(fontsize=8)

            # Plot MZI2
            if tolerance_val == 0.0:
                axes7[1, tol_idx].plot(t_ps, np.abs(resp_2_tol_norm), 'g-', linewidth=2, label='MZI2 Nominale')
            else:
                axes7[1, tol_idx].plot(t_ps, np.abs(responses_data_tolerance[param_type][0.0]['mzi2']), 'g:', alpha=0.5, linewidth=1, label='Nominale')
                axes7[1, tol_idx].plot(t_ps, np.abs(resp_2_tol_norm), 'r-', linewidth=2, label=f'±{tolerance_val}%')

            axes7[1, tol_idx].set_title(f'MZI2: {tolerance_val}%', fontsize=12)
            axes7[1, tol_idx].set_ylabel('|Ampiezza| MZI2')
            axes7[1, tol_idx].grid(True, alpha=0.3)
            if tol_idx == 0: axes7[1, tol_idx].legend(fontsize=8)

            # Plot MZI3
            if tolerance_val == 0.0:
                axes7[2, tol_idx].plot(t_ps, np.abs(resp_3_tol_norm), 'm-', linewidth=2, label='MZI3 Nominale')
            else:
                axes7[2, tol_idx].plot(t_ps, np.abs(responses_data_tolerance[param_type][0.0]['mzi3']), 'm:', alpha=0.5, linewidth=1, label='Nominale')
                axes7[2, tol_idx].plot(t_ps, np.abs(resp_3_tol_norm), 'r-', linewidth=2, label=f'±{tolerance_val}%')

            axes7[2, tol_idx].set_title(f'MZI3: {tolerance_val}%', fontsize=12)
            axes7[2, tol_idx].set_ylabel('|Ampiezza| MZI3')
            axes7[2, tol_idx].set_xlabel('Tempo (ps)')
            axes7[2, tol_idx].grid(True, alpha=0.3)
            if tol_idx == 0: axes7[2, tol_idx].legend(fontsize=8)

            responses_data_tolerance[param_type][tolerance_val] = {
                'mzi1': resp_1_tol_norm, 'mzi2': resp_2_tol_norm, 'mzi3': resp_3_tol_norm
            }

        plt.tight_layout()
        plt.savefig(f'studio5_tolleranze_{param_type}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Analisi deviazioni per questo parametro
        df_tolerance = analyze_tolerance_impact_and_save(responses_data_tolerance, tolerance_percent_list, [param_type])

        # Grafico deviazioni per questo parametro
        df_tol_rms = df_tolerance[df_tolerance['metric'] == 'rms'].copy()

        if len(df_tol_rms) > 0:
            fig8, axes8 = plt.subplots(1, 1, figsize=(10, 6))
            fig8.suptitle(f'DEVIAZIONI {param_type.upper()} - RMS Error', fontsize=14)

            for mzi_type in ['MZI1', 'MZI2', 'MZI3']:
                data_mzi = df_tol_rms[df_tol_rms['mzi_type'] == mzi_type]
                if len(data_mzi) > 0:
                    axes8.plot(data_mzi['tolerance_percent'], data_mzi['deviation_percent'], 
                              'o-', linewidth=2, markersize=6, label=mzi_type)

            axes8.set_xlabel('Tolleranza (%)')
            axes8.set_ylabel('Deviazione Percentuale (%)')
            axes8.set_title(f'Impatto Tolleranze {param_type.upper()}')
            axes8.grid(True, alpha=0.3)
            axes8.legend()
            axes8.set_yscale('log')

            plt.tight_layout()
            plt.savefig(f'studio5_deviazioni_{param_type}.png', dpi=300, bbox_inches='tight')
            plt.show()

    print("\n=== ANALISI COMPLETE ===")
    print("STUDIO 1: Geometrici vs Puri - Detuning")
    print("STUDIO 2: Geometrici vs Puri - Tau Variabile")
    print("STUDIO 3: Confronto Geometrici vs Puri")
    print("STUDIO 4: Deviazioni Geometrici vs Ideali")
    print("STUDIO 5: Tolleranze REALI e VISIBILI")
    
    return responses_data_geom, responses_data_pure, responses_data_tau_geom, responses_data_tau_pure, df_corrected

if __name__ == "__main__":
    results = main()
    data_geom, data_pure, data_tau_geom, data_tau_pure, df_corrected = results
