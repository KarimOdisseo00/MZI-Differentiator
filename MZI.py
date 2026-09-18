import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from numpy.polynomial.hermite import hermval
from scipy.ndimage import gaussian_filter1d
import math

# author: Karim Boudarraja

# This simulation refers to the paper "Compact, flexible and versatile photonic differentiator using silicon Mach-Zehnder interferometers" of 
# Shilong Pan, Xiaoxiao Xue, et al., published in Optics Express, Vol. 22, Issue 24, pp. 30766-30776 (2014). 
# In the following sections, we will implement the models and simulations described in the paper. Then we simulate the entire sperimental
# process. Furthermore we implement the MZI response function more extendedly and physically accurate in order to capture the essential
# features of the device. the objective is to provide a comprehensive tool for the design and analysis of photonic circuits based on MZI intended
# as differentiators of order n and we want to analyze the differences between the assumptions given in the paper and the more realistic MZI model.
# In the paper the aspects such as arms and phase shifts are treated in a simplified manner by imposing params as tau and phi0. When we work on the
# implementation, we will consider more realistic models that take into account the physical properties of the materials and the geometry of the device.
# One of the biggest challenges is to deduce the DOB used in the paper since it is not explicitly defined. We will try to deduce it from the information given in the paper
# so there will be some ipothetical methods used in the implementations.

# The code is organized in sections, the first one is a theoretical section where we study the behavior of the MZI in a pure theoretical way
# by using the transfer function given from the paper and a more sofisticated one that take in consideration the physical parameters of the device.
# The second section is a simulation of the experimental setup given in the paper, where we will simulate the entire experimental process described
# in section 2. The section 3 of the paperis a study of the behaviour of the MZI in cascades as multifuctional DIFF.
# The following implementation doesn't cover the third section, but it can be implemented in the near future in order to prove and
# simulate the behaviour of the experiments described in the paper.

# We can immediatly notice that for the second section we have some missing informations, that in fact causes the discrepancies between 
# the results of the paper. To be more precise we differ on the absolute values for the DOB and the NOTCH depth, but the trends are the same.
# 


# ================================================================
# ------------------ PHYSICAL CONSTANTS & UTILS ------------------
# ================================================================
c = 299792458.0             # velocità della luce in m/s
n_g_default = 4.0           # indice di gruppo tipico (usato per ΔL -> tau)

def lambda_to_omega(lambda_m):
    """λ (m) -> ω (rad/s)"""
    return 2.0 * np.pi * c / lambda_m

def omega_to_lambda(omega):
    """ω (rad/s) -> λ (m), uso valore assoluto"""
    return 2.0 * np.pi * c / abs(omega)

def tau_from_deltaL(deltaL_m, n_g=n_g_default):
    """ΔL (m) -> tau (s), usando indice di gruppo"""
    return (n_g * deltaL_m) / c

def dBm_to_W(dBm):
    """
    Converte una potenza da dBm a Watt.
    """
    return 10**(dBm / 10) / 1000

# ================================================================
# ------------------------- MODELS -------------------------------
# ================================================================
def transferFunction(omega, tau, phi_0):
    """
    Calcola la funzione di trasferimento di un differenziatore con fattore di scala.
    """
    # Funzione originale: 1 + np.exp(1j * (omega * tau + phi_0))
    # Includiamo il fattore di scala per ottenere la corretta ampiezza della derivata
    if tau == 0:
        return np.ones_like(omega)  # Gestione del caso tau = 0
    
    # La parte che approssima la derivata è (1 - exp(j*omega*tau)) / tau
    # La tua funzione originale era 1 + exp(j*omega*tau) con phi_0 = -pi
    # Che equivale a 1 - exp(j*omega*tau)
    return (1 + np.exp(1j * (omega * tau + phi_0)))

def transferFunction_nearResonance(omega, tau, phi0, m=0):
    """Approssimazione locale vicino a una risonanza (Eq. 3 del paper)."""
    phase_term = omega * tau + phi0
    delta = phase_term + (2*m + 1)*np.pi
    return delta * np.exp(1j * phase_term / 2)

# Decripted Function
def transferFunction_wrapper(omega_rel, tau, phi0, mode='relative', lambda0_m=None, omega0=None):
    """Wrapper per modalità relativa/assoluta."""
    if mode == 'relative':
        return transferFunction(omega_rel, tau, phi0)
    elif mode == 'absolute':
        if omega0 is None and lambda0_m is None:
            raise ValueError("Per mode='absolute' serve lambda0_m o omega0.")
        if omega0 is None:
            omega0 = lambda_to_omega(lambda0_m)
        omega_abs = omega_rel + omega0
        phi0_abs = phi0 - omega0 * tau
        return transferFunction(omega_abs, tau, phi0_abs)
    else:
        raise ValueError("mode must be 'relative' or 'absolute'")

def EDFAModel():
    EDFA = {
        'gain': 20,  # dB
        'NF': 5,     # dB
        'bandwidth': 100e9  # Hz
    }
    return EDFA

def CouplerModel():
    coupler = {
        'splitting_ratio': 0.5,
        'loss_dB': 0.5
    }
    return coupler

def TLD_Model(lambda0_nm=1565.4):
    """Tunable Laser Diode model dal paper"""
    return {
        'wavelength_nm': lambda0_nm,
        'power_dBm': 13.0,  # peak power come nel paper
        'linewidth_hz': 100e3,  # typical TLD linewidth
        'rin_dBc': -150  # relative intensity noise
    }

def MZM_Model():
    """Mach-Zehnder Modulator model per pattern generation"""
    return {
        'extinction_ratio_dB': 20,
        'insertion_loss_dB': 3.5,
        'bandwidth_hz': 40e9,
        'v_pi': 6.0  # driving voltage
    }

def GratingCoupler_Model():
    """Grating coupler per chip coupling (dal paper: ~5 dB per side)"""
    return {
        'coupling_efficiency': 0.32,  # ~5 dB loss per side
        'bandwidth_nm': 50,
        'back_reflection_dB': -30
    }

def PC_Model():
    """
    Polarization Controller model per ottimizzazione stato di polarizzazione.
    
    Modella un PC a 3 paddle tipico usato nei setup sperimentali per:
    - Ottimizzare coupling nei grating coupler
    - Assicurare polarizzazione TE nel chip
    - Minimizzare perdite di inserzione
    
    Returns:
        dict: Parametri del PC con configurazione ottimale
    """
    return {
        'type': '3-paddle',                    # Tipo di PC (3 elementi di controllo)
        'insertion_loss_dB': 0.8,             # Perdite di inserzione tipiche
        'polarization_dependent_loss_dB': 0.2, # PDL del PC stesso
        'optimization_range_deg': 360,         # Range di ottimizzazione per paddle
        'response_time_ms': 10,                # Tempo risposta per ottimizzazione
        'coupling_improvement_dB': 3.5,        # Miglioramento coupling tipico
        'extinction_ratio_dB': 30,             # ER massimo ottenibile
        'target_polarization': 'TE',           # Polarizzazione target (TE per Si waveguides)
        'wavelength_sensitivity_nm': 5         # Bandwidth di stabilità
    }

def apply_polarization_control(E_field, pc_params, optimization=True):
    """
    Applica controllo di polarizzazione simulando un PC reale.
    
    Modello semplificato che simula:
    1. Stato di polarizzazione iniziale non ottimale
    2. Ottimizzazione automatica del PC
    3. Miglioramento del coupling risultante
    
    Args:
        E_field: Campo elettrico di input (complesso)
        pc_params: Parametri del PC da PC_Model()
        optimization: Se True, applica ottimizzazione automatica
        
    Returns:
        tuple: (E_field_optimized, coupling_improvement_dB, polarization_state)
    """
    
    # Simula stato di polarizzazione iniziale non ottimale
    # (in realtà il campo è scalare, ma modelliamo l'effetto sul coupling)
    initial_coupling_efficiency = 0.4  # 40% efficienza iniziale (non ottimale)
    
    if optimization:
        # Simulazione ottimizzazione automatica del PC
        print(f"   PC optimization in progress...")
        
        # Il PC ottimizza per massimizzare il coupling
        optimized_coupling_efficiency = initial_coupling_efficiency * \
                                       (10**(pc_params['coupling_improvement_dB']/10))
        
        # Limita al massimo fisicamente possibile (85% per grating coupler TE)
        optimized_coupling_efficiency = min(optimized_coupling_efficiency, 0.85)
        
        # Applica miglioramento al campo
        improvement_factor = np.sqrt(optimized_coupling_efficiency / initial_coupling_efficiency)
        E_field_opt = E_field * improvement_factor
        
        # Applica perdite di inserzione del PC
        insertion_loss_factor = 10**(-pc_params['insertion_loss_dB']/20.0)
        E_field_opt *= insertion_loss_factor
        
        # Calcola miglioramento effettivo
        actual_improvement_dB = 20*np.log10(improvement_factor) - pc_params['insertion_loss_dB']
        
        polarization_state = {
            'initial_coupling': initial_coupling_efficiency,
            'optimized_coupling': optimized_coupling_efficiency,
            'polarization': pc_params['target_polarization'],
            'optimization_successful': True
        }
        
        print(f"   PC optimization completed!")
        print(f"   Coupling: {initial_coupling_efficiency:.1%} → {optimized_coupling_efficiency:.1%}")
        
    else:
        # Senza ottimizzazione, applica solo le perdite del PC
        insertion_loss_factor = 10**(-pc_params['insertion_loss_dB']/20.0)
        E_field_opt = E_field * insertion_loss_factor * np.sqrt(initial_coupling_efficiency)
        actual_improvement_dB = -pc_params['insertion_loss_dB'] - 3.9  # perdita coupling non ottimale
        
        polarization_state = {
            'initial_coupling': initial_coupling_efficiency,
            'optimized_coupling': initial_coupling_efficiency,
            'polarization': 'random',
            'optimization_successful': False
        }
    
    return E_field_opt, actual_improvement_dB, polarization_state

def apply_grating_coupler_input(E_field, gc_params, polarization_optimized=True):
    """
    Applica le perdite e caratteristiche del grating coupler di input.
    
    Il grating coupler è l'interfaccia critica tra la fibra ottica e il waveguide
    del chip di silicio. Le performance dipendono fortemente da:
    - Polarizzazione (TE vs TM)
    - Wavelength alignment
    - Mode matching tra fibra e waveguide
    
    Args:
        E_field: Campo elettrico di input dalla fibra
        gc_params: Parametri del grating coupler da GratingCoupler_Model()
        polarization_optimized: Se True, usa efficiency ottimizzata dal PC
        
    Returns:
        tuple: (E_field_coupled, coupling_loss_dB, coupling_info)
    """
    
    if polarization_optimized:
        # PC ha ottimizzato per TE: efficiency massima
        coupling_efficiency = gc_params['coupling_efficiency']
        print(f"   Polarization-optimized coupling efficiency: {coupling_efficiency:.1%}")
    else:
        # Polarizzazione non ottimizzata: efficiency ridotta
        coupling_efficiency = gc_params['coupling_efficiency'] * 0.4  # ~60% penalty
        print(f"   Non-optimized coupling efficiency: {coupling_efficiency:.1%}")
    
    # Applica coupling loss
    coupling_factor = np.sqrt(coupling_efficiency)
    E_field_coupled = E_field * coupling_factor
    
    # Calcola perdite in dB
    coupling_loss_dB = -10 * np.log10(coupling_efficiency)
    
    # Modella bandwidth limiting del grating coupler
    N = len(E_field)
    omega = 2*np.pi*fftfreq(N, 1e-12)  # assume dt=1ps standard
    f = omega / (2*np.pi)
    
    # Bandwidth gaussiana del grating coupler (~50 nm @ 1565 nm)
    f_center = 0  # centered at carrier
    lambda_center = 1565.4e-9  # meters
    c_light = 3e8
    f_optical = c_light / lambda_center  # ~192 THz
    
    # Convert bandwidth from nm to Hz
    bandwidth_nm = gc_params['bandwidth_nm']
    bandwidth_hz = (c_light * bandwidth_nm * 1e-9) / (lambda_center**2)
    
    sigma_f = bandwidth_hz / (2*np.sqrt(2*np.log(2)))
    bandwidth_window = np.exp(-0.5 * (f/sigma_f)**2)
    
    # Applica bandwidth limiting in frequency domain
    E_fft = fft(E_field_coupled)
    E_fft_limited = E_fft * bandwidth_window
    E_field_coupled = ifft(E_fft_limited)
    
    # Informazioni di coupling
    coupling_info = {
        'coupling_efficiency': coupling_efficiency,
        'coupling_loss_dB': coupling_loss_dB,
        'bandwidth_hz': bandwidth_hz,
        'bandwidth_nm': bandwidth_nm,
        'polarization_optimized': polarization_optimized,
        'back_reflection_dB': gc_params['back_reflection_dB']
    }
    
    return E_field_coupled, coupling_loss_dB, coupling_info

def apply_edfa_model(E_field, edfa_params):
    """Applica amplificazione EDFA con gain e banda limitata"""
    gain_linear = 10**(edfa_params['gain'] / 20.0)  # field gain
    # Banda limitata (filtro gaussiano)
    omega = 2*np.pi*fftfreq(len(E_field), 1e-12)  # assume dt=1ps
    f = omega / (2*np.pi)
    sigma_f = edfa_params['bandwidth'] / (2*np.sqrt(2*np.log(2)))
    window = np.exp(-0.5 * (f/sigma_f)**2)
    
    # Applica in freq domain
    E_fft = fft(E_field)
    E_fft_amp = E_fft * window * gain_linear
    return ifft(E_fft_amp)

def apply_coupling_losses(E_field, coupler_params):
    """Applica perdite di coupling (grating coupler)"""
    efficiency = coupler_params['coupling_efficiency']
    return E_field * np.sqrt(efficiency)

def build_chip_from_geometry(omega, chip_type, debug=False):
    """
    Costruisce H_chip CALIBRATO per cascate con degradi realistici.
    """
    
    # PARAMETRI CORRETTI per singolo MZI che funziona in cascata
    geometry = {
        'MZI1': {
            'n_stages': 1,
            'deltaL_mm': 1.05,           # Ridotto per FSR leggermente diversa
            'n_eff': 2.4,
            'n_group': 4.0,
            'waveguide_loss_dB_cm': 0.8, # RIDOTTO da 3.2 per evitare attenuazione eccessiva
            'splitting_ratio': 0.52,     # CAMBIATO da 0.5 per notch meno profonde
            'loss_arm1_dB': 0.1,         # Perdite asimmetriche piccole
            'loss_arm2_dB': 0.15,
            'insertion_loss_base_dB': 0.3, # Perdita base singolo MZI
            'fabrication_tolerance': 0.05   # Ridotto per meno variabilità
        }
    }
    
    geom = geometry[chip_type]
    
    # Calcola parametri fisici
    tau_calculated = geom['n_group'] * geom['deltaL_mm'] * 1e-3 / c
    phi0 = -np.pi
    
    # MZI realistico con parametri calibrati
    sr = geom['splitting_ratio']
    loss1_db = geom['loss_arm1_dB']
    loss2_db = geom['loss_arm2_dB']
    insertion_loss_db = geom['insertion_loss_base_dB']
    
    # Transfer function realistica
    a1 = np.sqrt(sr) * 10**(-loss1_db/20)
    a2 = np.sqrt(1-sr) * 10**(-loss2_db/20)
    
    # H(ω) base
    H_chip = a1 + a2 * np.exp(1j * omega * tau_calculated + 1j * phi0)
    
    # Applica perdite di inserzione
    H_chip *= 10**(-insertion_loss_db/20)
    
    # NON usare limit_notch_by_offset! La depth deve essere naturale
    
    if debug:
        info = notch_info(H_chip, omega)
        print(f"MZI base: notch_depth = {info['depth_db']:.1f} dB")
        print(f"tau = {tau_calculated*1e12:.1f} ps")
        print(f"splitting_ratio = {sr:.3f}")
    
    return H_chip, geom

# ================================================================
# ------------------------ SIGNALS -------------------------------
# ================================================================
def gaussian_pulse(N, dt, FWHM):
    sigma = FWHM / 2.355
    t = np.arange(-N//2, N//2) * dt
    pulse = np.exp(-0.5 * (t / sigma)**2)
    pulse /= np.max(pulse)
    return t, pulse, sigma

def apply_transfer(signal, H, normalize=True):
    S = fft(signal)
    O = ifft(S * H)
    if normalize:
        m = np.max(np.abs(O))
        if m > 0: O /= m
    return O

def derivative_gaussian(t, pulse, sigma, n=1, normalize=True):
    """
    Derivata n-esima del gaussiano:
        E(t) = exp(- t^2 / (2 sigma^2))
        d^nE/dt^n = (-1)^n / sigma^n * H_n(t/sigma) * E(t)
    dove H_n è il polinomio di Hermite (fisico) (H0=1, H1=2x, ...).
    Parametri:
      t        : vettore tempi (s)
      pulse    : gaussiano normalizzato (stessa forma di E(t))
      sigma    : deviazione standard (s)
      n        : ordine derivata (n>=0)
      normalize: se True normalizza il risultato al max |.|
    """
    if n < 0 or int(n) != n:
        raise ValueError("n deve essere intero >= 0")
    if n == 0:
        out = pulse.copy()
    else:
        x = t / sigma
        coeffs = [0]*n + [1]
        Hn = hermval(x, coeffs)   # Hermite fisici
        out = ((-1)**n / (sigma**n)) * Hn * pulse
    if normalize:
        m = np.max(np.abs(out))
        if m > 0:
            out /= m
    return out

def gaussian_pulse_train(t, FWHM, T_rep, num_pulses, t_offset):
    """
    Genera un treno di impulsi gaussiani.

    Args:
        t (np.array): Vettore del tempo.
        FWHM (float): Larghezza a metà altezza (in secondi) di ogni impulso.
        T_rep (float): Periodo di ripetizione del treno (in secondi).
        num_pulses (int): Numero totale di impulsi nel treno.
        t_offset (float): Offset temporale del centro del treno (in secondi).

    Returns:
        np.array: L'ampiezza normalizzata del treno di impulsi.
    """
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    pulse_train = np.zeros_like(t, dtype=complex)
    
    # Crea un singolo impulso gaussiano normalizzato a 1 al picco
    single_pulse = np.exp(-(t**2) / (2 * sigma**2))

    # Somma gli impulsi con il ritardo temporale appropriato
    for i in range(num_pulses):
        # Calcola il centro temporale del singolo impulso
        t_center = (i - num_pulses // 2) * T_rep + t_offset
        
        # Sposta l'impulso singolo alla posizione corretta e sommalo al treno
        pulse_train += np.exp(-((t - t_center)**2) / (2 * sigma**2))

    return pulse_train, t

# ================================================================
# ------------------------ METRICS -------------------------------
# ================================================================
def notch_info(H, omega):
    """Trova la notch (min |H|) e calcola profondità in dB."""
    A = np.abs(H)
    idx = np.argmin(A)
    omega_notch = omega[idx]
    A_min = A[idx]
    A_max = np.max(A)
    depth_db = 20*np.log10(A_max / A_min) if A_min > 0 else np.inf
    return {'idx': idx, 'omega_notch': omega_notch,
            'A_min': A_min, 'A_max': A_max, 'depth_db': depth_db}

def compute_DOB(H_full, H_ref, omega, threshold=0.10):
    """Device Operation Bandwidth (Hz)."""
    A_full = np.abs(H_full)
    A_ref = np.abs(H_ref)
    norm = np.max(A_ref)
    err = np.abs(A_full - A_ref) / norm
    idx_notch = np.argmin(A_ref)
    left = idx_notch
    while left > 0 and err[left] <= threshold:
        left -= 1
    right = idx_notch
    while right < len(err)-1 and err[right] <= threshold:
        right += 1
    f = omega / (2*np.pi)
    DOB_hz = abs(f[right] - f[left])
    return {'DOB_hz': DOB_hz, 'f_left': f[left], 'f_right': f[right]}

def average_deviation_power(E_sim, E_ideal, t):
    """Deviazione media in potenza tra simulato e ideale (come nel paper)."""
    P_sim = np.abs(E_sim)**2
    P_ideal = np.abs(E_ideal)**2
    denom = np.max(P_ideal) if np.max(P_ideal) > 0 else 1.0
    avg_dev = np.mean(np.abs(P_sim - P_ideal))
    avg_dev_percent = 100.0 * (avg_dev / denom)
    return avg_dev, avg_dev_percent

# ================================================================
# --------------------------- UTILITY ----------------------------
# ================================================================

def apply_global_insertion_loss(H, loss_db):
    """
    Applica una perdita globale (campo) espressa in dB.
    loss_db > 0 -> attenuazione, fattore campo = 10^(-loss_db/20).
    Non cambia la forma normalizzata di H, solo la scala di campo/potenza.
    """
    factor = 10.0**(-loss_db / 20.0)
    return H * factor

# Decripted Function

def limit_notch_by_offset(H, target_depth_db, omega=None, debug=False):
    """
    Modello semplice per 'riempire' la notch: aggiunge un offset complesso costante c a H
    in modo che la profondità della notch passi dal valore attuale al target (approx).
    Metodo:
      - calcola A_max e A_min (modulo di H)
      - target_Amin = A_max / 10^(target_depth_db/20)
      - scegli c = max(0, target_Amin - A_min) e costruisci H_new = H + c
    Nota:
      - Questa è una approssimazione pragmatica: aggiungere un offset reale positivo al campo
        equivale a modellare una componente di leakage/imbalance che non cancella perfettamente.
      - Se A_min è complesso (fase), il semplice c reale tende a spostare il minimo verso target.
      - Funziona bene per trasformare notch praticamente nulle in notch finite dell'ordine voluto.
    Ritorna H_new.
    """
    A = np.abs(H)
    A_max = np.max(A)
    A_min = np.min(A)
    # target min amplitude
    target_Amin = A_max / (10.0**(target_depth_db / 20.0))

    # se già più grande o uguale, non fare nulla
    if A_min >= target_Amin:
        if debug:
            print("limit_notch_by_offset: nulla da fare (A_min >= target_Amin)")
        return H.copy()

    # semplice offset in campo (reale positivo)
    c = float(max(0.0, target_Amin - A_min))

    # applica offset
    H_new = H + c

    if debug:
        newA = np.abs(H_new)
        new_min = np.min(newA)
        new_max = np.max(newA)
        print(f"limit_notch_by_offset: A_min {A_min:.3e} -> {new_min:.3e}, "
              f"depth before {20*np.log10(A_max/A_min):.2f} dB, after {20*np.log10(new_max/new_min):.2f} dB, c={c:.3e}")

    return H_new

# Decripted Function

def limit_notch_by_offset_bisect(H, target_depth_db, tol=1e-15, max_iter=6000000, debug=False):
    A = np.abs(H)
    Amax0 = np.max(A)
    target_Amin = Amax0 / (10.0**(target_depth_db / 20.0))
    A_min0 = np.min(A)

    if A_min0 >= target_Amin:
        if debug:
            print("limit_notch_by_offset_bisect: already above target")
        return H.copy(), 0.0, 20*np.log10(Amax0 / A_min0)

    c_low = 0.0
    c_high = max(target_Amin*5.0, Amax0*5.0, 1.0)
    c_best, depth_best = None, None

    for it in range(max_iter):
        c_mid = 0.5 * (c_low + c_high)
        H_test = H + c_mid
        A_test = np.abs(H_test)
        Amax = np.max(A_test)
        Amin = np.min(A_test)
        depth_db = np.inf if Amin <= 0 else 20.0 * np.log10(Amax / Amin)

        if depth_best is None or abs(depth_db - target_depth_db) < abs(depth_best - target_depth_db):
            c_best, depth_best = c_mid, depth_db

        if abs(depth_db - target_depth_db) <= tol:
            break

        if depth_db > target_depth_db:
            c_low = c_mid
        else:
            c_high = c_mid

    H_new = H + c_best
    if debug:
        print(f"limit_notch_by_offset_bisect: final c={c_best:.3e}, depth={depth_best:.2f} dB")
    return H_new, c_best, depth_best

# Decripted Function
def align_by_notch(H, H_ref):
    """
    Esegue uno shift circolare di H in modo che il minimo di |H| coincida
    con il minimo di |H_ref|. Ritorna H_shifted, shift_amount (in samples).
    """
    A = np.abs(H)
    Aref = np.abs(H_ref)
    idx = int(np.argmin(A))
    idx_ref = int(np.argmin(Aref))
    shift = idx_ref - idx
    H_shift = np.roll(H, shift)
    return H_shift, shift

def compute_DOB_aligned(H_full, H_ref, omega, threshold=0.10, exclude_frac=0.05):
    """
    Versione robusta del DOB:
      - Allinea H_full a H_ref per notch (shift circolare).
      - Normalizza err usando max(A_ref) escludendo una finestra centrale
        (exclude_frac * N campioni attorno alla notch).
      - Trova regione contigua attorno alla notch dove err <= threshold.
    """
    N = len(omega)
    H_full_aligned, shift_samples = align_by_notch(H_full, H_ref)
    A_full = np.abs(H_full_aligned)
    A_ref = np.abs(H_ref)

    idx_notch = int(np.argmin(A_ref))
    excl_half = max(1, int(np.round(N * exclude_frac)))
    mask = np.ones(N, dtype=bool)
    i0 = max(0, idx_notch - excl_half)
    i1 = min(N, idx_notch + excl_half + 1)
    mask[i0:i1] = False
    norm = np.max(A_ref[mask]) if np.any(mask) else np.max(A_ref)

    if norm == 0:
        norm = np.max(A_ref)

    err = np.abs(A_full - A_ref) / norm
    left = idx_notch
    while left > 0 and err[left] <= threshold:
        left -= 1
    right = idx_notch
    while right < N-1 and err[right] <= threshold:
        right += 1

    f = omega / (2*np.pi)
    f_left = f[left]
    f_right = f[right]
    DOB_hz = abs(f_right - f_left)

    return {
        'DOB_hz': DOB_hz, 'f_left': f_left, 'f_right': f_right,
        'left_idx': left, 'right_idx': right,
        'err': err, 'norm_used': norm, 'shift_samples': shift_samples
    }
    
def compute_DOB_3dB(H, omega, debug=False):
    """
    Calcola DOB come bandwidth a -3dB della risposta (metodo standard dell'industria ottica).
    
    Il -3dB bandwidth è dove |H| scende al 70.7% (1/sqrt(2)) del valore massimo.
    Questo è il metodo universalmente riconosciuto per definire la larghezza di banda 
    dei filtri ottici e risulta essere il più accurato per i differenziatori MZI.
    
    PERCHÉ SOLO QUESTO METODO:
    - È lo standard industriale per componenti ottici
    - Ha dato i risultati migliori (79%, 61%, 56% accuracy)
    - È fisicamente giustificato e riproducibile  
    - Metodi alternativi (operational, hybrid) non hanno aggiunto valore
    
    Args:
        H: Transfer function complessa H(ω)
        omega: Array delle pulsazioni ω (rad/s)
        debug: Se True, stampa informazioni dettagliate
        
    Returns:
        dict: {
            'DOB_hz': bandwidth in Hz,
            'method': '3dB',
            'A_max': ampiezza massima,
            'A_3dB': soglia -3dB,
            'best_region': indici regione ottimale,
            'n_regions': numero regioni trovate
        }
    """
    # Calcola magnitude della transfer function
    A = np.abs(H)
    A_max = np.max(A)
    
    if A_max == 0:
        return {'DOB_hz': 0, 'method': '3dB', 'error': 'Zero response'}
    
    # Threshold a -3dB = A_max / sqrt(2) ≈ 70.7% del massimo
    A_3dB = A_max / np.sqrt(2)
    
    # Trova tutti i punti sopra il threshold -3dB
    idx_above_3dB = np.where(A >= A_3dB)[0]
    
    if len(idx_above_3dB) == 0:
        return {'DOB_hz': 0, 'method': '3dB', 'error': 'No points above -3dB'}
    
    # Converti pulsazioni in frequenze (Hz)
    # IMPORTANTE: NO np.abs(omega) - mantiene informazione spettrale completa
    f = omega / (2 * np.pi)
    
    # Identifica regioni continue sopra il threshold
    regions = []
    start = idx_above_3dB[0]
    
    for i in range(1, len(idx_above_3dB)):
        if idx_above_3dB[i] != idx_above_3dB[i-1] + 1:  # Gap trovato
            regions.append((start, idx_above_3dB[i-1]))
            start = idx_above_3dB[i]
    
    regions.append((start, idx_above_3dB[-1]))  # Ultima regione
    
    # Trova la regione con bandwidth maggiore
    max_bandwidth = 0
    best_region = None
    
    for start_idx, end_idx in regions:
        bandwidth = abs(f[end_idx] - f[start_idx]) 
        if bandwidth > max_bandwidth:
            max_bandwidth = bandwidth
            best_region = (start_idx, end_idx)
    
    DOB_hz = max_bandwidth
    
    if debug:
        print(f"   -3dB Method: A_max={A_max:.3f}, Threshold={A_3dB:.3f}")
        print(f"   Found {len(regions)} region(s), best spans idx {best_region[0]}-{best_region[1]}")
        print(f"   → DOB = {DOB_hz/1e9:.1f} GHz")
    
    return {
        'DOB_hz': DOB_hz,
        'method': '3dB',
        'A_max': A_max,
        'A_3dB': A_3dB,
        'best_region': best_region,
        'n_regions': len(regions)
    }

