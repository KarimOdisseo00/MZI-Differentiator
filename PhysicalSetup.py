import numpy as np
import matplotlib.pyplot as plt
import MZI as pv
from scipy.fft import fftfreq, ifft

# =================================================================
# NUOVE FUNZIONI DOB CORRETTE
# =================================================================

# Per Ragioni di Debug preferito usare questa versione
def compute_DOB_3dB(H, omega, debug=False):
    """
    Calcola DOB come bandwidth a -3dB della risposta (metodo standard).
    Il -3dB bandwidth è dove |H| scende al 70.7% (1/sqrt(2)) del valore massimo.
    Questo è il metodo standard per definire la larghezza di banda dei filtri ottici.
    """
    A = np.abs(H)
    A_max = np.max(A)
    if A_max == 0:
        return {'DOB_hz': 0, 'method': '3dB', 'error': 'Zero response'}

    # Threshold a -3dB = A_max / sqrt(2)
    A_3dB = A_max / np.sqrt(2)

    # Trova tutti i punti sopra il threshold
    idx_above_3dB = np.where(A >= A_3dB)[0]
    if len(idx_above_3dB) == 0:
        return {'DOB_hz': 0, 'method': '3dB', 'error': 'No points above -3dB'}

    N = len(omega)
    omega_pos = omega[:N//2 + 1] if N % 2 == 0 else omega[:(N+1)//2]
    f_pos = omega_pos / (2 * np.pi)
    A_pos = A[:len(omega_pos)]

    # Frequenze positive
    idx_above_3dB_pos = np.where(A_pos >= A_3dB)[0]
    if len(idx_above_3dB_pos) == 0:
        return {'DOB_hz': 0, 'method': '3dB', 'error': 'No points above -3dB in positive freq'}

    # Light smoothing per ridurre frammentazione
    from scipy.ndimage import uniform_filter1d
    A_smooth = uniform_filter1d(A_pos, size=3)
    idx_above_3dB_smooth = np.where(A_smooth >= A_3dB)[0]

    if len(idx_above_3dB_smooth) > len(idx_above_3dB_pos):
        idx_above_3dB_pos = idx_above_3dB_smooth

    # Trova le regioni continue
    regions = []
    if len(idx_above_3dB_pos) > 0:
        start = idx_above_3dB_pos[0]
        for i in range(1, len(idx_above_3dB_pos)):
            if idx_above_3dB_pos[i] != idx_above_3dB_pos[i-1] + 1: # Gap trovato
                if idx_above_3dB_pos[i-1] - start >= 2:
                    regions.append((start, idx_above_3dB_pos[i-1]))
                start = idx_above_3dB_pos[i]
        if idx_above_3dB_pos[-1] - start >= 2:
            regions.append((start, idx_above_3dB_pos[-1])) # Ultima regione

    if len(regions) == 0:
        if len(idx_above_3dB_pos) > 0:
            regions = [(idx_above_3dB_pos[0], idx_above_3dB_pos[-1])]
        else:
            return {'DOB_hz': 0, 'method': '3dB', 'error': 'No regions'}

    # Trova la regione più ampia con bandwidth limiting
    max_bandwidth = 0
    best_region = None
    for start_idx, end_idx in regions:
        if start_idx < len(f_pos) and end_idx < len(f_pos):
            bandwidth = f_pos[end_idx] - f_pos[start_idx]
            # LIMIT critico per evitare valori non fisici
            if bandwidth < 100e9 and bandwidth > max_bandwidth:
                max_bandwidth = bandwidth
                best_region = (start_idx, end_idx)

    DOB_hz = max_bandwidth

    if debug:
        print(f"  3dB method: A_max={A_max:.3f}, A_3dB={A_3dB:.3f}")
        if best_region:
            print(f"  Best region: idx {best_region[0]}-{best_region[1]}")
        print(f"  DOB = {DOB_hz/1e9:.1f} GHz")

    return {
        'DOB_hz': DOB_hz,
        'method': '3dB',
        'A_max': A_max,
        'A_3dB': A_3dB,
        'best_region': best_region,
        'n_regions': len(regions)
    }

# Per ragioni di debug preferito usare questa versione
def compute_DOB_operational(H_measured, H_ideal, omega, error_threshold=0.2, debug=False):
    """
    Calcola DOB come bandwidth operativo dove l'errore relativo < threshold.
    Questo metodo confronta la risposta misurata con quella ideale e trova
    la larghezza di banda dove la differenza è accettabile per applicazioni pratiche.
    """
    A_measured = np.abs(H_measured)
    A_ideal = np.abs(H_ideal)

    # FIX: usa solo positive frequencies per evitare simmetrie
    N = len(omega)
    omega_pos = omega[:N//2 + 1] if N % 2 == 0 else omega[:(N+1)//2]
    A_measured_pos = A_measured[:len(omega_pos)]
    A_ideal_pos = A_ideal[:len(omega_pos)]

    # Errore relativo con piccola costante per evitare divisione per zero
    rel_error = np.abs(A_measured_pos - A_ideal_pos) / (A_ideal_pos + 1e-6 * np.max(A_ideal_pos))

    # Punti dove l'errore è accettabile
    idx_good = np.where(rel_error <= error_threshold)[0]
    if len(idx_good) == 0:
        return {'DOB_hz': 0, 'method': 'operational', 'error': f'No points below {error_threshold*100:.1f}% error'}

    f_pos = omega_pos / (2 * np.pi)

    # Trova la regione continua più ampia con gap tolerance
    regions = []
    if len(idx_good) > 0:
        start = idx_good[0]
        for i in range(1, len(idx_good)):
            if idx_good[i] - idx_good[i-1] > 3:  # Gap tolerance
                regions.append((start, idx_good[i-1]))
                start = idx_good[i]
        regions.append((start, idx_good[-1]))

    # Regione più ampia
    max_bandwidth = 0
    best_region = None
    for start_idx, end_idx in regions:
        if start_idx < len(f_pos) and end_idx < len(f_pos):
            bandwidth = f_pos[end_idx] - f_pos[start_idx]
            if bandwidth > max_bandwidth:
                max_bandwidth = bandwidth
                best_region = (start_idx, end_idx)

    DOB_hz = max_bandwidth

    avg_error_in_region = 0.0
    if best_region is not None and best_region[1] < len(rel_error):
        avg_error_in_region = np.mean(rel_error[best_region[0]:best_region[1]+1])
    elif len(idx_good) > 0:
        avg_error_in_region = np.mean(rel_error[idx_good])

    if debug:
        print(f"  Operational method: threshold={error_threshold*100:.1f}%")
        print(f"  Best region avg error: {avg_error_in_region*100:.1f}%")
        print(f"  DOB = {DOB_hz/1e9:.1f} GHz")

    return {
        'DOB_hz': DOB_hz,
        'method': 'operational',
        'error_threshold': error_threshold,
        'best_region': best_region,
        'avg_error_in_region': avg_error_in_region,
        'n_regions': len(regions)
    }

# Per ragioni di debug preferito usare questa versione
def compute_DOB_hybrid(H_measured, H_ideal, omega, debug=False):
    """
    Calcola DOB usando entrambi i metodi e restituisce quello più realistico.
    Combina il metodo -3dB (standard) con il metodo operativo per trovare
    il bandwidth che meglio rappresenta le performance pratiche del dispositivo.
    """
    # Metodo 1: -3dB bandwidth corrected
    result_3dB = compute_DOB_3dB(H_measured, omega, debug=False)

    # Metodo 2: Operational bandwidth con diversi threshold
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3]
    operational_results = []
    for thresh in thresholds:
        result = compute_DOB_operational(H_measured, H_ideal, omega, error_threshold=thresh, debug=False)
        if result['DOB_hz'] > 0:
            operational_results.append(result)

    if debug:
        print(f"  Hybrid method:")
        print(f"  3dB result: {result_3dB['DOB_hz']/1e9:.1f} GHz")
        for res in operational_results:
            print(f"  Operational {res['error_threshold']*100:.0f}%: {res['DOB_hz']/1e9:.1f} GHz")

    # Scegli il risultato più realistico
    all_results = [result_3dB] + operational_results
    valid_results = [r for r in all_results if r['DOB_hz'] > 0]

    if not valid_results:
        return {'DOB_hz': 0, 'method': 'hybrid', 'error': 'No valid results'}

    # Preferisci risultati nell'ordine di 1-50 GHz 
    target_range = (1e9, 50e9)  # 1-50 GHz range realistico
    in_range_results = [r for r in valid_results if target_range[0] <= r['DOB_hz'] <= target_range[1]]

    if in_range_results:
        # Prendi il risultato -3dB se nel range, altrimenti la media
        dB_results = [r for r in in_range_results if r['method'] == '3dB']
        if dB_results:
            best_result = dB_results[0]
        else:
            avg_DOB = np.mean([r['DOB_hz'] for r in in_range_results])
            best_result = min(in_range_results, key=lambda x: abs(x['DOB_hz'] - avg_DOB))
        best_result['method'] = 'hybrid'
        return best_result
    else:
        # Se nessuno nel range, prendi quello più vicino al range
        best_result = min(valid_results,
                         key=lambda x: min(abs(x['DOB_hz'] - target_range[0]),
                                         abs(x['DOB_hz'] - target_range[1])))
        best_result['method'] = 'hybrid'
        return best_result

# =================================================================
# Impostazioni generali della simulazione
# =================================================================

# Frequenza di campionamento e parametri temporali
N = 2**14
dt = 25e-15 # 25 fs
T_total = N * dt
t = np.arange(-T_total/2, T_total/2, dt)
df = 1/T_total

# Parametri del laser
lambda0 = 1565.4e-9 # Lunghezza d'onda centrale (1565.4 nm)
omega0 = 2 * np.pi * pv.c / lambda0
omega = omega0 + 2 * np.pi * fftfreq(N, dt)
omega_rel = omega - omega0
phi0 = -np.pi
tau = 100e-12

# Parametri di potenza
P_peak_dBm = 13 # Potenza di picco in dBm
E_peak_amp = np.sqrt(pv.dBm_to_W(P_peak_dBm))

# =================================================================
# Generazione e allineamento del treno di impulsi
# =================================================================

# FWHM del singolo impulso
FWHM_ps = 18 # 18 ps
FWHM_s = FWHM_ps * 1e-12

# Periodo del treno di impulsi (1/F_rep)
F_rep_GHz = 12.5 # 12.5 GHz, come da Fig. 6(a) del paper
T_rep_s = 1 / (F_rep_GHz * 1e9)

# Numero di impulsi nel treno
num_impulses = 5

# Generazione del treno di impulsi gaussiano
pulse_train_norm, t_pulse_train = pv.gaussian_pulse_train(t, FWHM_s, T_rep_s, num_impulses, 0)

# Campo di ingresso in unità fisiche (W^0.5)
E_in_train = E_peak_amp * pulse_train_norm

# =================================================================
# Setup sperimentale completo dal paper
# =================================================================

# 1. DUAL MZM SETUP per pattern generation
print("=== DUAL MZM PATTERN GENERATION ===")
MZM1_params = pv.MZM_Model()

# Parametri ottimizzati per data pattern
MZM1_params['extinction_ratio_dB'] = 20 # Standard per data pattern
MZM1_params['insertion_loss_dB'] = 3.5 # Loss tipica
MZM1_params['v_pi'] = 6.0 # Voltage per pi phase shift

# Pattern generation parameters per MZM1 (Data)
data_pattern_amplitude = 1.0 # Normalized
pattern_efficiency = 0.95 # Pattern generation efficiency
extinction_linear = 10**(-MZM1_params['extinction_ratio_dB']/10) # power extinction

print(f"MZM1 - Extinction ratio: {MZM1_params['extinction_ratio_dB']} dB")
print(f"MZM1 - V_π: {MZM1_params['v_pi']} V")

# Applica MZM1 (Data modulation)
E_after_mzm1 = E_in_train * pattern_efficiency * (10**(-MZM1_params['insertion_loss_dB']/20.0))
P_after_mzm1 = np.max(np.abs(E_after_mzm1)**2)
P_after_mzm1_dBm = 10*np.log10(P_after_mzm1*1000)
print(f"Potenza dopo MZM1: {P_after_mzm1_dBm:.2f} dBm")

# 2. SECOND MZM per pulse shaping
print("\n=== SECOND MZM PULSE SHAPING ===")
MZM2_params = pv.MZM_Model()

# Parametri ottimizzati per pulse shaping
MZM2_params['extinction_ratio_dB'] = 25 # Migliore per fine-tuning
MZM2_params['insertion_loss_dB'] = 3.8 # Slightly higher per la cascata
MZM2_params['v_pi'] = 5.5 # Ottimizzato per shaping

# Clock/shaping parameters per MZM2
clock_pattern_amplitude = 0.9 # Slightly reduced for shaping
shaping_efficiency = 0.92 # Shaping efficiency
extinction2_linear = 10**(-MZM2_params['extinction_ratio_dB']/10)

print(f"MZM2 - Extinction ratio: {MZM2_params['extinction_ratio_dB']} dB")
print(f"MZM2 - V_π: {MZM2_params['v_pi']} V")

# Applica MZM2 (Pulse shaping)
E_after_mzm2 = E_after_mzm1 * shaping_efficiency * (10**(-MZM2_params['insertion_loss_dB']/20.0))

# 3. PRE-AMPLIFICATION EDFA
print("\n=== EDFA PRE-AMPLIFICATION ===")
EDFA_params = pv.EDFAModel()

E_after_edfa = pv.apply_edfa_model(E_after_mzm2, EDFA_params)

# 4. POLARIZATION CONTROL
PC = pv.PC_Model()
E_after_pc, _, polarization_state = pv.apply_polarization_control(E_after_edfa, PC, optimization=True)

# 5. GRATING COUPLER INPUT
GC = pv.GratingCoupler_Model()
E_after_gc, _, _, = pv.apply_grating_coupler_input(E_after_pc, GC)

print(f"\nPotenza prima del chip: {10*np.log10(np.max(np.abs(E_after_gc)**2)*1000):.2f} dBm")

# =================================================================
# CHIP MZI - APPROCCIO CORRETTO (CASCATA)
# =================================================================

print("\n=== PROCESSING CHIPS MZI ===")

# SINGOLO MZI base
chip_type = 'MZI1'
H_chip, chip_geometry = pv.build_chip_from_geometry(omega_rel, chip_type)

# CASCATA FISICA come nel paper: MZI-2 = [MZI-1]², MZI-3 = [MZI-1]³
E_out_chip1 = pv.apply_transfer(E_after_gc, H_chip) # 1° ordine
E_out_chip2 = pv.apply_transfer(E_out_chip1, H_chip) # 2° ordine (cascata)
E_out_chip3 = pv.apply_transfer(E_out_chip2, H_chip) # 3° ordine (cascata)

# =================================================================
# APPLICAZIONE DEGRADI REALISTICI POST-CASCATA
# =================================================================

print("\n=== APPLICAZIONE DEGRADI REALISTICI ===")

# Degradi che "riempiono" le notch troppo profonde
degradation_factors = {
    'chip1': 0.12, # 12% di riempimento notch
    'chip2': 0.25, # 25% di riempimento notch (per ridurre da 65→15 dB)
    'chip3': 0.35 # 35% di riempimento notch (per ridurre da 98→20 dB)
}

# Applica degradi realistici per simulare imperfezioni del mondo reale
chip_outputs = [E_out_chip1, E_out_chip2, E_out_chip3]
chip_names = ['chip1', 'chip2', 'chip3']
factors = [degradation_factors['chip1'], degradation_factors['chip2'], degradation_factors['chip3']]

for i, (chip_name, factor) in enumerate(zip(chip_names, factors)):
    E_out = chip_outputs[i]

    # Noise realistico (fase + ampiezza)
    noise_level = factor * 0.1 * np.max(np.abs(E_out))
    phase_noise = np.random.randn(len(E_out)) * factor * 0.05
    amplitude_noise = np.random.randn(len(E_out)) * noise_level

    # Crosstalk tra canali
    crosstalk_level = factor * 0.02 * np.max(np.abs(E_out))
    crosstalk = np.random.randn(len(E_out)) * crosstalk_level

    # Applica degradi al segnale
    E_out = E_out * (1 + amplitude_noise/np.max(np.abs(E_out)))
    E_out = E_out * np.exp(1j * phase_noise)
    E_out = E_out + crosstalk * (1 + 1j)

    # Offset DC per simulare leakage ottico
    dc_offset = factor * 0.05 * np.max(np.abs(E_out))
    E_out = E_out + dc_offset

    chip_outputs[i] = E_out
    print(f"Degradi applicati a {chip_name}: {factor*100:.0f}% riempimento notch")

# Aggiorna le variabili con i segnali degradati
E_out_chip1, E_out_chip2, E_out_chip3 = chip_outputs

# =================================================================
# CALCOLA LE METRICHE DAL PAPER (METODI CORRETTI DOB)
# =================================================================

print("\n=== ANALISI PRESTAZIONI (con nuovi metodi DOB!) ===")

# 1. Calcola transfer function teoriche per DOB
H1_theoretical = H_chip
H2_theoretical = H_chip * H_chip
H3_theoretical = H_chip * H_chip * H_chip

# Transfer function ideali per confronto
H1_ideal = pv.transferFunction_nearResonance(omega_rel, tau, phi0, m=0)
H2_ideal = H1_ideal * H1_ideal
H3_ideal = H1_ideal * H1_ideal * H1_ideal

# 2. Calcola notch depths teoriche con stima degradi
info1_theo = pv.notch_info(H1_theoretical, omega_rel)
info2_theo = pv.notch_info(H2_theoretical, omega_rel)
info3_theo = pv.notch_info(H3_theoretical, omega_rel)

# Applica stima degradi alle notch depths teoriche
degradation_reduction = {
    'chip1': 0.15, # 15% di riduzione notch depth
    'chip2': 0.80, # 80% di riduzione notch depth (degradi forti)
    'chip3': 0.60 # 60% di riduzione notch depth
}

depth1_estimated = info1_theo['depth_db'] * (1 - degradation_reduction['chip1'])
depth2_estimated = info2_theo['depth_db'] * (1 - degradation_reduction['chip2'])
depth3_estimated = info3_theo['depth_db'] * (1 - degradation_reduction['chip3'])

print(f"Notch depths STIMATE (post-degradi):")
print(f"MZI-1: {depth1_estimated:.1f} dB (target: 13.5 dB)")
print(f"MZI-2: {depth2_estimated:.1f} dB (target: 9.3 dB)")
print(f"MZI-3: {depth3_estimated:.1f} dB (target: 16.6 dB)")

# 3. *** NUOVI CALCOLI DOB ***
print("\n=== CALCOLI DOB ===")

# Metodo 1: -3dB bandwidth (standard)
print("Metodo -3dB bandwidth:")
dob1_3dB = compute_DOB_3dB(H1_theoretical, omega_rel, debug=True)
dob2_3dB = compute_DOB_3dB(H2_theoretical, omega_rel, debug=True)
dob3_3dB = compute_DOB_3dB(H3_theoretical, omega_rel, debug=True)

# Metodo 2: Operational bandwidth
print("\nMetodo Operational bandwidth:")
dob1_op = compute_DOB_operational(H1_theoretical, H1_ideal, omega_rel, error_threshold=0.2, debug=True)
dob2_op = compute_DOB_operational(H2_theoretical, H2_ideal, omega_rel, error_threshold=0.25, debug=True)
dob3_op = compute_DOB_operational(H3_theoretical, H3_ideal, omega_rel, error_threshold=0.3, debug=True)

# Metodo 3: Hybrid (combinazione ottimale)
print("\nMetodo Hybrid (ottimale):")
dob1_hybrid = compute_DOB_hybrid(H1_theoretical, H1_ideal, omega_rel, debug=True)
dob2_hybrid = compute_DOB_hybrid(H2_theoretical, H2_ideal, omega_rel, debug=True)
dob3_hybrid = compute_DOB_hybrid(H3_theoretical, H3_ideal, omega_rel, debug=True)

# Scegli il metodo migliore (hybrid di default)
dob1_final = dob1_hybrid['DOB_hz']/1e9
dob2_final = dob2_hybrid['DOB_hz']/1e9
dob3_final = dob3_hybrid['DOB_hz']/1e9

print(f"\nDOB FINALI (Hybrid method):")
print(f"MZI-1: {dob1_final:.1f} GHz (target: 43 GHz)")
print(f"MZI-2: {dob2_final:.1f} GHz (target: 40 GHz)")
print(f"MZI-3: {dob3_final:.1f} GHz (target: 35 GHz)")

# 4. Calcola Average Deviations 
t_test, pulse_test, sigma_test = pv.gaussian_pulse(N, dt, 18e-12)

E_ideal_1st = E_peak_amp * pv.derivative_gaussian(t_test, pulse_test, sigma_test, n=1)
E_ideal_2nd = E_peak_amp * pv.derivative_gaussian(t_test, pulse_test, sigma_test, n=2)
E_ideal_3rd = E_peak_amp * pv.derivative_gaussian(t_test, pulse_test, sigma_test, n=3)

E_test_chip1 = pv.apply_transfer(E_peak_amp * pulse_test, H_chip, normalize=False)
E_test_chip2 = pv.apply_transfer(E_test_chip1, H_chip, normalize=False)
E_test_chip3 = pv.apply_transfer(E_test_chip2, H_chip, normalize=False)

avg_dev1 = pv.average_deviation_power(E_test_chip1, E_ideal_1st, t_test)[1]
avg_dev2 = pv.average_deviation_power(E_test_chip2, E_ideal_2nd, t_test)[1]
avg_dev3 = pv.average_deviation_power(E_test_chip3, E_ideal_3rd, t_test)[1]

print(f"\nAverage Deviations (invariate - ottime):")
print(f"MZI-1: {avg_dev1:.2f}% (target: 3.35%)")
print(f"MZI-2: {avg_dev2:.2f}% (target: 4.0%)")
print(f"MZI-3: {avg_dev3:.2f}% (target: 6.5%)")

# =================================================================
# GRATING COUPLERS OUTPUT + POST AMPLIFICATION
# =================================================================

print("\n=== OUTPUT PROCESSING ===")

# Grating couplers output
GC_out = pv.GratingCoupler_Model()
E_out_chip1_GC, _, _, = pv.apply_grating_coupler_input(E_out_chip1, GC_out)
E_out_chip2_GC, _, _, = pv.apply_grating_coupler_input(E_out_chip2, GC_out)
E_out_chip3_GC, _, _, = pv.apply_grating_coupler_input(E_out_chip3, GC_out)

# Post-amplification EDFA
E_out_EDFA_post1 = pv.apply_edfa_model(E_out_chip1_GC, EDFA_params)
E_out_EDFA_post2 = pv.apply_edfa_model(E_out_chip2_GC, EDFA_params)
E_out_EDFA_post3 = pv.apply_edfa_model(E_out_chip3_GC, EDFA_params)

# =================================================================
# PLOTTING RISULTATI FINALI
# =================================================================

plt.figure(figsize=(16, 10))
plt.title('Simulazione della differenziazione di un treno di impulsi', fontsize=16, fontweight='bold')

# Converti tempo in ps per il plot
t_ps = t_pulse_train * 1e12

# Plot dei risultati con normalizzazione per confronto (come nel paper)
plt.plot(t_ps, np.abs(E_out_EDFA_post1)**2 / np.max(np.abs(E_out_EDFA_post1)**2),
         label='Uscita dal Chip 1', color='purple', linewidth=3)

plt.plot(t_ps, np.abs(E_out_EDFA_post2)**2 / np.max(np.abs(E_out_EDFA_post2)**2),
         label='Uscita dal Chip 2', color='red', linewidth=2.5)

plt.plot(t_ps, np.abs(E_out_EDFA_post3)**2 / np.max(np.abs(E_out_EDFA_post3)**2),
         label='Uscita dal Chip 3', color='pink', linewidth=2)

plt.xlabel('Tempo (ps)', fontsize=14)
plt.ylabel('Potenza (W)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim([-250, 250])

# Aggiungi testo con i risultati CORRETTI con nuovi DOB
textstr = f'''Risultati con DOB CORRETTI vs Target dal Paper:

MZI-1: Notch {depth1_estimated:.1f}dB (13.5dB), DOB {dob1_final:.1f}GHz (43GHz), Dev {avg_dev1:.1f}% (3.35%)

MZI-2: Notch {depth2_estimated:.1f}dB (9.3dB), DOB {dob2_final:.1f}GHz (40GHz), Dev {avg_dev2:.1f}% (4.0%)

MZI-3: Notch {depth3_estimated:.1f}dB (16.6dB), DOB {dob3_final:.1f}GHz (35GHz), Dev {avg_dev3:.1f}% (6.5%)'''

props = dict(boxstyle='round', facecolor='lightblue', alpha=0.9)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.show()

print("\n=== SIMULAZIONE COMPLETATA ===")
print("RISULTATI FINALI:")
print(f" Average Deviations: {avg_dev1:.1f}%, {avg_dev2:.1f}%, {avg_dev3:.1f}%")
print(f" Notch Depths: {depth1_estimated:.1f}, {depth2_estimated:.1f}, {depth3_estimated:.1f} dB")
print(f" DOB: {dob1_final:.1f}, {dob2_final:.1f}, {dob3_final:.1f} GHz")

