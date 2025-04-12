# 🚀 Walter's Military-Grade Breathing Condensate Simulation
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Parameters ---
n_points = 200            # Grid size (1D)
kappa = 1.0               # Breathing stiffness
Lambda = 0.7              # Energy scale
g_coupling = 5.0          # Strong neighbor coupling (relational force)
t_max = 500               # Long simulation time
n_times = 2000            # Number of time points

# --- Breathing Field Potential ---
def breathing_force(phi):
    return Lambda**4 * np.sin(phi)

# --- System of ODEs ---
def breathing_system(t, y):
    phi = y[:n_points]
    dphi = y[n_points:]
    ddphi = np.zeros_like(phi)

    # Apply breathing dynamics + neighbor coupling
    for i in range(n_points):
        left = phi[i-1] if i > 0 else phi[i]
        right = phi[i+1] if i < n_points-1 else phi[i]
        neighbor_force = g_coupling * (left + right - 2 * phi[i])
        ddphi[i] = (breathing_force(phi[i]) + neighbor_force) / kappa

    return np.concatenate([dphi, ddphi])

# --- Initial Conditions: full chaos (random between -π and π) ---
np.random.seed(42)  # For reproducibility
phi0 = np.random.uniform(-np.pi, np.pi, n_points)
dphi0 = np.random.uniform(-0.1, 0.1, n_points)  # Tiny random initial velocities
y0 = np.concatenate([phi0, dphi0])

# --- Time Grid ---
t_eval = np.linspace(0, t_max, n_times)

# --- Solve the system ---
sol = solve_ivp(breathing_system, [0, t_max], y0, t_eval=t_eval, method='RK45')

# --- Reshape solution ---
phi_sol = sol.y[:n_points, :]  # Breathing field evolution over time

# --- Plot as heatmap ---
plt.figure(figsize=(14, 7))
plt.imshow(phi_sol, extent=[0, t_max, 0, n_points], aspect='auto', cmap='plasma', origin='lower')
plt.colorbar(label='Breathing Field ϕ')
plt.title("Walter's Spartan Breathing Proto-Matter Simulation 🛡️🌌")
plt.xlabel('Time')
plt.ylabel('Position on 1D Grid')
plt.show()
