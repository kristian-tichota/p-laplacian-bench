import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Parameters ---
p = 3.0             # Exponent p > 2 (modifiable)
h = 1.0             # Dirichlet boundary condition u(t, 0) = h
T = 0.05             # Final simulation time
N = 500             # Increased spatial resolution for accuracy
eps = 1e-3          # Regularization parameter to prevent singular diffusion
threshold = 1e-3    # Wavefront boundary cutoff

# --- Spatial Grid ---
x = np.linspace(0, 1, N)
dx = x[1] - x[0]

# --- ODE System (Method of Lines) ---
def p_laplacian_flux(t, u_inner):
    """Computes the regularized p-Laplacian spatial derivatives."""
    u = np.zeros(N)
    u[0] = h
    u[1:-1] = u_inner
    u[-1] = 0.0
    
    # Forward difference for spatial gradients
    du_dx = np.diff(u) / dx
    
    # Regularized flux: (|u_x|^2 + eps^2)^((p-2)/2) * u_x
    flux = (du_dx**2 + eps**2)**((p - 2) / 2) * du_dx
    
    # Divergence of the flux
    du_dt = np.diff(flux) / dx
    
    return du_dt

# --- Time Integration ---
u0 = np.zeros(N - 2)
t_eval = np.linspace(1e-4, T, 300)

# Integrate stiff system using LSODA
solution = solve_ivp(p_laplacian_flux, [0, T], u0, method='LSODA', t_eval=t_eval)

# --- Wavefront Tracking with Sub-grid Interpolation ---
zeta_numerical = np.zeros_like(t_eval)

for i, t in enumerate(t_eval):
    u_t = np.zeros(N)
    u_t[0] = h
    u_t[1:-1] = solution.y[:, i]
    u_t[-1] = 0.0
    
    # Identify domain exceeding the threshold
    active_domain = np.where(u_t > threshold)[0]
    
    if len(active_domain) > 0:
        idx = active_domain[-1]
        
        # Apply sub-grid linear interpolation if not at the right boundary
        if idx < N - 1:
            u0, u1 = u_t[idx], u_t[idx+1]
            fraction = (threshold - u0) / (u1 - u0)
            zeta_numerical[i] = x[idx] + fraction * dx
        else:
            zeta_numerical[i] = x[idx]

# --- Analytical Bounds ---
alpha = (p - 1) / (p - 2)
K = (h**(p - 2)) * (alpha**(p - 1))

zeta_lower = (K * t_eval)**(1 / p)
zeta_upper = (p / (p - 1)) * (K * (p - 1) * t_eval)**(1 / p)

# --- Visualization (Color-blind Friendly) ---
# Okabe-Ito color palette for maximum accessibility
color_num = '#000000'    # Black
color_lower = '#56B4E9'  # Sky Blue
color_upper = '#D55E00'  # Vermillion

plt.figure(figsize=(9, 6))

# Applying distinct line styles along with the accessible palette
plt.plot(t_eval, zeta_numerical, color=color_num, linestyle='-', linewidth=2.5, label=r'Numerical Wavefront $\zeta(t)$')
plt.plot(t_eval, zeta_lower, color=color_lower, linestyle=':', linewidth=3.0, label='Analytical Lower Bound')
plt.plot(t_eval, zeta_upper, color=color_upper, linestyle='--', linewidth=2.5, label='Analytical Upper Bound')

plt.xlabel('Time $t$', fontsize=12)
plt.ylabel('Wavefront Position $x$', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='-', alpha=0.3)
plt.xlim(0, T)
plt.ylim(0, min(1.0, np.max(zeta_upper) * 1.1))
plt.tight_layout()

# --- File Export ---
plt.savefig('envelope.pdf', format='pdf', bbox_inches='tight', transparent=True)
