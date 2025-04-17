import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmaz, sigmax, sigmay, mesolve, Bloch, expect

def simulate_rabi_oscillations(omega, delta, gamma, t_max, n_points, pulse_start, pulse_duration):
    # Initial state |0⟩
    psi0 = basis(2, 0)
    
    # Time points
    times = np.linspace(0, t_max, n_points)
    
    # Instead of using a function directly, we'll use the list-string format
    # that QuTiP provides for time-dependent Hamiltonians
    
    # Define the time-independent terms
    H0 = (delta/2) * sigmaz()  # Always present (free evolution)
    H1 = (omega/2) * sigmax()  # Only during pulse
    
    # Define the coefficient function for H1
    def pulse_envelope(t, args):
        # Check if time t is within the pulse window
        if pulse_start <= t <= (pulse_start + pulse_duration):
            return 1.0
        else:
            return 0.0
    
    # Create the full Hamiltonian list
    H = [H0, [H1, pulse_envelope]]
    
    # Collapse operators for decay/decoherence
    c_ops = []
    if gamma > 0:
        # Relaxation - T1 decay
        c_ops.append(np.sqrt(gamma) * (basis(2, 0) * basis(2, 1).dag()))
        # Dephasing - T2 decay
        c_ops.append(np.sqrt(gamma/2) * sigmaz())
    
    # Solve the system with time-dependent Hamiltonian
    result = mesolve(H, psi0, times, c_ops, [sigmaz(), sigmax(), sigmay()])
    
    return times, result.expect


def app():
    st.title("Rabi Oscillations Simulator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        omega = st.slider("Rabi Frequency (Ω)", 0.0, 10.0, 1.0)
        delta = st.slider("Detuning (Δ)", -5.0, 5.0, 0.0)
        
    with col2:
        gamma = st.slider("Decay Rate (γ)", 0.0, 1.0, 0.0)
        t_max = st.slider("Simulation Time", 1.0, 30.0, 10.0)
    
    # Add pulse parameters to main interface
    st.subheader("Excitation Pulse Parameters")
    pulse_start = st.slider("Pulse Start Time", 0.0, t_max, 0.0)
    pulse_duration = st.slider("Pulse Duration", 0.0, 10.0, 1.0)
    
    times, expects = simulate_rabi_oscillations(omega, delta, gamma, t_max, 200, pulse_start, pulse_duration)
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, expects[0], label='⟨σz⟩')
    ax.plot(times, expects[1], label='⟨σx⟩')
    ax.plot(times, expects[2], label='⟨σy⟩')
    
    # Highlight the pulse duration on the plot
    ax.axvspan(pulse_start, pulse_start + pulse_duration, alpha=0.2, color='red', label='Pulse')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Expectation Value')
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)
    
    # Add Bloch sphere visualization at specific times
    if st.checkbox("Show Bloch Sphere Evolution"):
        selected_times = st.slider("Select time points", 
                                  min_value=float(times[0]), 
                                  max_value=float(times[-1]), 
                                  value=(float(times[0]), float(times[-1])))
        
        # Calculate Bloch sphere coordinates
        x = expects[1]  # <σx>
        y = expects[2]  # <σy>
        z = expects[0]  # <σz>
        
        # Create figure with two subplots
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Slice times for the selected range
        mask = (times >= selected_times[0]) & (times <= selected_times[1])
        times_slice = times[mask]
        x_slice = x[mask]
        y_slice = y[mask]
        z_slice = z[mask]

        # Plot Bloch sphere coordinates over time
        ax1.plot(times_slice, x_slice, label='x')
        ax1.plot(times_slice, y_slice, label='y')
        ax1.plot(times_slice, z_slice, label='z')
        ax1.axvspan(pulse_start, pulse_start + pulse_duration, alpha=0.2, color='red', label='Pulse')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Bloch Coordinates')
        ax1.legend()
        ax1.grid(True)

        # Create Bloch sphere at selected time
        selected_idx = np.interp(selected_times[1], times, range(len(times))) # Correct
        selected_idx = int(selected_idx) # To int because is a index

        ax2 = fig2.add_subplot(1, 2, 2, projection='3d')
        b = Bloch(axes=ax2)

        # Add the entire trajectory in the selected time range
        b.add_points([x_slice, y_slice, z_slice])

        # Highlight the final point with a vector
        b.add_vectors([x[selected_idx], y[selected_idx], z[selected_idx]])
        b.render()

        st.pyplot(fig2)


def simulate_ramsey(delta, t_wait, gamma, n_points):
    # Time points for waiting time
    times = np.linspace(0, t_wait, n_points)
    
    # First pi/2 pulse around X - creates superposition
    psi0 = basis(2, 0)
    H_pulse = (np.pi/2) * sigmax()
    result_pulse = mesolve(H_pulse, psi0, [0, 0.5], [], [])
    psi_after_pulse = result_pulse.states[-1]
    
    # Free evolution with detuning
    H_free = (delta/2) * sigmaz()
    
    # Collapse operators for decay/decoherence
    c_ops = []
    if gamma > 0:
        # Pure dephasing dominates in Ramsey experiments
        c_ops.append(np.sqrt(gamma) * sigmaz())
    
    # Solve the system during free evolution with decay
    result = mesolve(H_free, psi_after_pulse, times, c_ops, [sigmaz(), sigmax(), sigmay()])
    
    return times, result.expect

def ramsey_app():
    st.title("Ramsey Interference Simulator")
    
    delta = st.slider("Detuning (Δ)", -5.0, 5.0, 1.0)
    t_wait = st.slider("Maximum Wait Time", 0.1, 20.0, 10.0)
    gamma = st.slider("Dephasing Rate (γ)", 0.0, 1.0, 0.1)
    
    times, expects = simulate_ramsey(delta, t_wait, gamma, 200)
    
    # Plot the interference pattern
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, expects[0], label='Population (⟨σz⟩)')
    ax.plot(times, expects[1], label='⟨σx⟩')
    ax.plot(times, expects[2], label='⟨σy⟩')
    ax.set_xlabel('Wait Time')
    ax.set_ylabel('Expectation Value')
    ax.set_title('Ramsey Interference Pattern')
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)

def simulate_photon_echo(delta_distribution, t_max, t_pulse, gamma, n_points):
    # Time points for simulation
    times = np.linspace(0, t_max, n_points)
    
    # Arrays to store average signals
    avg_sx = np.zeros(n_points)
    avg_sy = np.zeros(n_points)
    avg_sz = np.zeros(n_points)
    
    # Number of ensemble members with different detunings
    n_ensemble = 30
    detunings = np.random.normal(0, delta_distribution, n_ensemble)
    
    # Simulate each qubit in the ensemble
    for delta in detunings:
        # Start in ground state
        psi0 = basis(2, 0)
        
        # Apply π/2 pulse (instantaneously for simplicity)
        # This creates a superposition state along y-axis
        psi = (basis(2, 0) + 1j*basis(2, 1))/np.sqrt(2)
        
        # Calculate state at each time point
        for i, t in enumerate(times):
            # Calculate current state
            if t < t_pulse:
                # Free evolution before π-pulse
                phase = delta * t
                # Add dephasing
                coherence = np.exp(-gamma * t) if gamma > 0 else 1.0
                
                # State with accumulated phase and decoherence
                psi_t = (basis(2, 0) + coherence * np.exp(1j * phase) * 1j * basis(2, 1))
                psi_t = psi_t / psi_t.norm()
                
            else:
                # After π-pulse: phase reversal
                # Time elapsed since π-pulse
                t_since_pulse = t - t_pulse
                # Total phase: accumulated before pulse, then reversal
                phase = delta * (t_pulse - t_since_pulse)
                # Continuous decoherence
                coherence = np.exp(-gamma * t) if gamma > 0 else 1.0
                
                # State with reversed phase and decoherence
                psi_t = (basis(2, 0) + coherence * np.exp(1j * phase) * 1j * basis(2, 1))
                psi_t = psi_t / psi_t.norm()
            
            # Calculate expectation values
            avg_sx[i] += expect(sigmax(), psi_t) / n_ensemble
            avg_sy[i] += expect(sigmay(), psi_t) / n_ensemble
            avg_sz[i] += expect(sigmaz(), psi_t) / n_ensemble
    
    return times, (avg_sx, avg_sy, avg_sz)

def echo_app():
    st.title("Photon Echo Simulator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        delta_dist = st.slider("Detuning Distribution Width", 0.1, 5.0, 1.0)
        t_max = st.slider("Maximum Simulation Time", 5.0, 30.0, 15.0)
    
    with col2:
        t_pulse = st.slider("π-Pulse Time", 1.0, t_max-1.0, t_max/2)
        gamma = st.slider("Dephasing Rate (γ)", 0.0, 1.0, 0.1)
    
    times, (avg_sx, avg_sy, avg_sz) = simulate_photon_echo(delta_dist, t_max, t_pulse, gamma, 300)
    
    # Plot the echo pattern
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, avg_sx, label='⟨σx⟩')
    ax.plot(times, avg_sy, label='⟨σy⟩')
    ax.plot(times, avg_sz, label='⟨σz⟩')
    ax.axvline(x=t_pulse, linestyle='--', color='gray', label='π pulse')
    ax.set_xlabel('Time')
    ax.set_ylabel('Average Magnetization')
    ax.set_title('Photon Echo')
    ax.legend()
    ax.grid(True)
    
    # Add echo visibility calculation
    echo_time = 2 * t_pulse
    echo_index = np.argmin(np.abs(times - echo_time))
    if echo_index > 0:
        echo_amplitude = np.sqrt(avg_sx[echo_index]**2 + avg_sy[echo_index]**2)
        st.write(f"Echo Amplitude at t = {echo_time:.2f}: {echo_amplitude:.4f}")
    
    st.pyplot(fig)

def main():
    st.sidebar.title("Quantum Phenomena Simulator")
    app_selection = st.sidebar.radio(
        "Select Simulation",
        ["Rabi Oscillations", "Ramsey Interference [WIP]", "Photon Echo [WIP]"]
    )
    
    if app_selection == "Rabi Oscillations":
        app()
    elif app_selection == "Ramsey Interference":
        ramsey_app()
    elif app_selection == "Photon Echo":
        echo_app()

if __name__ == "__main__":
    main()