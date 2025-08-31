# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 18:56:04 2025

@author: gauta
"""

"""
NLS (focusing) with power-law nonlinearity in 1D:
    i ψ_t + 1/2 ψ_xx + |ψ|^{2σ} ψ = 0
Stationary ansatz ψ(x,t) = φ(x) e^{i β t} (β>0) yields the ODE
    -β φ + 1/2 φ'' + φ^{2σ+1} = 0,    φ(x) ≥ 0, φ → 0 as |x|→∞.

This script computes stationary soliton profiles φ(x) for given (β, σ)
using a finite-difference collocation on [-L, L] with Dirichlet BCs,and
Newton's method. It then builds the bifurcation curves:
    (i) peak amplitude A(β) = φ(0) vs β
    (ii) mass M(β) = ∫ |φ|^2 dx vs β (for VK criterion)

Dependencies: numpy, scipy, matplotlib
Tested with: Python ≥3.9, numpy ≥1.23, scipy ≥1.9, matplotlib ≥3.7
"""

#from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

# SciPy is preferred for sparse ops/solvers, but we provide
# a dense fallback if sparse is unavailable.
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from scipy.sparse import diags
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    sp = None
    spla = None
    diags = None


@dataclass
class SolverConfig:
    L: float = 20.0            # half-domain size; domain is [-L, L]
    N: int = 2000              # total grid points (including boundaries)
    newton_tol: float = 1e-10  # Newton residual tolerance
    newton_maxit: int = 40     # maximum Newton iterations
    line_search: bool = True   # backtracking line search for robustness
    ls_alpha: float = 1e-4     # Armijo parameter
    ls_beta: float = 0.5       # step shrink factor
    verbose: bool = False      # print Newton progress


def grid_and_laplacian_dirichlet(L: float, N: int):
    """Create a uniform grid on [-L, L] and the 1D second-derivative
    finite-difference matrix with Dirichlet BCs at both ends.

    Returns
    -------
    x : (N,) ndarray grid points
    D2_in : (n_in, n_in) sparse/dense matrix for interior points (excludes boundaries)
    dx : float grid spacing
    """
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]

    # Interior indices exclude boundaries 0 and N-1
    n_in = N - 2

    if SCIPY_AVAILABLE:
        main = -2.0 * np.ones(n_in)
        off = 1.0 * np.ones(n_in - 1)
        D2_in = diags([off, main, off], offsets=[-1, 0, 1], format='csc') / dx**2
    else:
        # Dense fallback
        D2_in = (-2.0 * np.eye(n_in) + np.eye(n_in, k=1) + np.eye(n_in, k=-1)) / dx**2

    return x, D2_in, dx


def initial_guess(x: np.ndarray, beta: float, sigma: float) -> np.ndarray:
    """Provide a smooth, positive, even initial guess for φ on the full grid,
    satisfying Dirichlet BC approximately. We use a Gaussian centered at 0.
    The width is tied to β so that larger β → narrower guess.
    """
    # Heuristic width ~ 1/sqrt(beta)
    width = max(1e-2, 1.0 / np.sqrt(max(beta, 1e-3)))
    A = (beta * (sigma + 1.0))**(1.0 / (2.0 * max(sigma, 1e-8)))  # rough scaling
    phi0 = A * np.exp(-(x/width)**2)
    # enforce Dirichlet at boundaries explicitly
    phi0[0] = 0.0
    phi0[-1] = 0.0
    return phi0


def stationary_profile(beta: float, sigma: float, cfg: SolverConfig) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
    """Solve for stationary φ(x) on [-L, L] given (β, σ) by Newton's method.

    We discretize on N points with Dirichlet BCs and solve for interior values
    φ_in. The residual vector for interior nodes i is:
        R_i = -β φ_i + 1/2 (D2 φ)_i + φ_i^{2σ+1}

    Returns
    -------
    x : grid (N,)
    phi : profile on the full grid including zeros at boundaries
    converged : bool
    info : dict with diagnostics (iterations, residual norms, step sizes)
    """
    x, D2_in, dx = grid_and_laplacian_dirichlet(cfg.L, cfg.N)
    # Interior slice indices
    i0, i1 = 1, cfg.N - 1

    # Initial guess (full grid) then restrict to interior
    phi_full = initial_guess(x, beta, sigma)
    phi = phi_full[i0:i1].copy()

    # Newton iteration
    hist = []
    converged = False

    for k in range(cfg.newton_maxit):
        # Nonlinearity and residual on interior points
        # guard against negative/complex by enforcing non-negativity during Newton (project)
        phi = np.maximum(phi, 0.0)
        if np.any(np.isnan(phi)) or np.any(~np.isfinite(phi)):
            break

        if SCIPY_AVAILABLE:
            D2phi = D2_in @ phi
        else:
            D2phi = D2_in.dot(phi)

        nonlin = phi ** (2.0 * sigma + 1.0)
        R = -beta * phi + 0.5 * D2phi + nonlin
        res_norm = float(np.linalg.norm(R, ord=2))
        hist.append(res_norm)
        if cfg.verbose:
            print(f"Newton iter {k}: ||R||_2 = {res_norm:.3e}")

        if res_norm < cfg.newton_tol:
            converged = True
            break

        # Build Jacobian: J = -β I + 0.5 D2 + (2σ+1) φ^{2σ} diag
        diag_nonlin = (2.0 * sigma + 1.0) * np.power(phi, 2.0 * sigma)
        if SCIPY_AVAILABLE:
            J = (-beta) * sp.eye(phi.size, format='csc') + 0.5 * D2_in + sp.diags(diag_nonlin, 0, format='csc')
            try:
                delta = spla.spsolve(J, -R)
            except Exception:
                # fall back to dense solve
                Jd = J.toarray()
                delta = np.linalg.solve(Jd, -R)
        else:
            J = (-beta) * np.eye(phi.size) + 0.5 * D2_in + np.diag(diag_nonlin)
            delta = np.linalg.solve(J, -R)

        # Line search (Armijo) to ensure decrease of residual norm
        step = 1.0
        phi_new = phi + step * delta
        if cfg.line_search:
            # simple backtracking based on residual norm
            def res_norm_for(p):
                p = np.maximum(p, 0.0)
                if SCIPY_AVAILABLE:
                    return float(np.linalg.norm(-beta * p + 0.5 * (D2_in @ p) + p ** (2.0 * sigma + 1.0)))
                else:
                    return float(np.linalg.norm(-beta * p + 0.5 * (D2_in.dot(p)) + p ** (2.0 * sigma + 1.0)))

            r0 = res_norm
            while step > 1e-6:
                rn = res_norm_for(phi_new)
                if rn <= (1.0 - cfg.ls_alpha * step) * r0:
                    break
                step *= cfg.ls_beta
                phi_new = phi + step * delta
        phi = phi_new

    # Assemble full solution with Dirichlet boundaries
    phi_full = np.zeros_like(x)
    phi_full[i0:i1] = np.maximum(phi, 0.0)

    info = {
        "iterations": k + 1,
        "residual_norm": hist[-1] if hist else np.inf,
        "history": hist,
        "dx": dx,
    }

    return x, phi_full, converged, info


def mass_M(x: np.ndarray, phi: np.ndarray) -> float:
    dx = x[1] - x[0]
    return float(np.trapz(phi**2, x))


def width_variance(x: np.ndarray, phi: np.ndarray) -> float:
    M = mass_M(x, phi)
    if M <= 0:
        return np.nan
    x0 = float(np.trapz(x * (phi**2), x) / M)
    W2 = float(np.trapz(((x - x0)**2) * (phi**2), x) / M)
    return W2


def branch_over_beta(beta_list: List[float], sigma: float, cfg: SolverConfig,
                     warm_start: bool = True):
    """Trace a solution branch by sweeping β values.

    If warm_start=True, use previous φ as initial guess for the next β to
    greatly improve convergence and continuity.
    """
    results = []
    phi_guess_full: Optional[np.ndarray] = None

    for j, beta in enumerate(beta_list):
        if warm_start and phi_guess_full is not None:
            # Build a guess by rescaling previous solution slightly
            x, _, _ = grid_and_laplacian_dirichlet(cfg.L, cfg.N)
            guess = phi_guess_full.copy()
            # small amplitude tweak to encourage convergence when β changes
            scale = (beta / beta_list[max(j-1, 0)])**0.25
            guess *= scale

            def stationary_with_guess(beta, sigma, cfg, guess_full):
                x, D2_in, dx = grid_and_laplacian_dirichlet(cfg.L, cfg.N)
                i0, i1 = 1, cfg.N - 1
                phi = guess_full[i0:i1].copy()

                hist = []
                converged = False
                for k in range(cfg.newton_maxit):
                    phi = np.maximum(phi, 0.0)
                    if SCIPY_AVAILABLE:
                        D2phi = D2_in @ phi
                    else:
                        D2phi = D2_in.dot(phi)
                    R = -beta * phi + 0.5 * D2phi + phi ** (2.0 * sigma + 1.0)
                    res_norm = float(np.linalg.norm(R))
                    hist.append(res_norm)
                    if res_norm < cfg.newton_tol:
                        converged = True
                        break
                    diag_nonlin = (2.0 * sigma + 1.0) * np.power(phi, 2.0 * sigma)
                    if SCIPY_AVAILABLE:
                        J = (-beta) * sp.eye(phi.size, format='csc') + 0.5 * D2_in + sp.diags(diag_nonlin, 0, format='csc')
                        try:
                            delta = spla.spsolve(J, -R)
                        except Exception:
                            delta = np.linalg.solve(J.toarray(), -R)
                    else:
                        J = (-beta) * np.eye(phi.size) + 0.5 * D2_in + np.diag(diag_nonlin)
                        delta = np.linalg.solve(J, -R)
                    step = 1.0
                    phi_new = phi + step * delta
                    if cfg.line_search:
                        def resn(p):
                            p = np.maximum(p, 0.0)
                            if SCIPY_AVAILABLE:
                                return float(np.linalg.norm(-beta * p + 0.5 * (D2_in @ p) + p ** (2.0 * sigma + 1.0)))
                            else:
                                return float(np.linalg.norm(-beta * p + 0.5 * (D2_in.dot(p)) + p ** (2.0 * sigma + 1.0)))
                        r0 = res_norm
                        while step > 1e-6:
                            rn = resn(phi_new)
                            if rn <= (1.0 - cfg.ls_alpha * step) * r0:
                                break
                            step *= cfg.ls_beta
                            phi_new = phi + step * delta
                    phi = phi_new

                phi_full = np.zeros(cfg.N)
                phi_full[0] = 0.0
                phi_full[-1] = 0.0
                phi_full[i0:i1] = np.maximum(phi, 0.0)

                info = {"iterations": k + 1, "residual_norm": hist[-1] if hist else np.inf, "history": hist}
                return x, phi_full, converged, info

            x, phi, conv, info = stationary_with_guess(beta, sigma, cfg, guess)
        else:
            x, phi, conv, info = stationary_profile(beta, sigma, cfg)

        A = float(np.max(phi))
        M = mass_M(x, phi)
        W2 = width_variance(x, phi)
        results.append({
            "beta": beta,
            "sigma": sigma,
            "amplitude": A,
            "mass": M,
            "width2": W2,
            "converged": conv,
            "info": info,
            "x": x,
            "phi": phi,
        })
        phi_guess_full = phi
        if cfg.verbose:
            print(f"β={beta:.4f}: A={A:.5f}, M={M:.5f}, converged={conv}, iters={info['iterations']}")

    return results


def plot_bifurcation(results_by_sigma: Dict[float, List[dict]], save_prefix: Optional[str] = None):
    """Plot A(β) and M(β) branches for each σ; optionally save figures."""
    # Amplitude vs beta
    plt.figure()
    for sigma, res in results_by_sigma.items():
        betas = [r["beta"] for r in res]
        amps = [r["amplitude"] for r in res]
        plt.plot(betas, amps, marker='o', label=f"σ={sigma}")
    plt.xlabel(r"β (propagation constant)")
    plt.ylabel(r"peak amplitude A = max φ")
    plt.title("Bifurcation curve: amplitude vs β")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_prefix:
        plt.savefig(f"{save_prefix}_A_vs_beta.png", dpi=160, bbox_inches='tight')

    # Mass vs beta (VK criterion uses dM/dβ sign)
    plt.figure()
    for sigma, res in results_by_sigma.items():
        betas = [r["beta"] for r in res]
        masses = [r["mass"] for r in res]
        plt.plot(betas, masses, marker='o', label=f"σ={sigma}")
    plt.xlabel(r"β (propagation constant)")
    plt.ylabel(r"Mass  M = ∫ φ^2 dx")
    plt.title("Branch mass M(β) — VK diagnostic")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_prefix:
        plt.savefig(f"{save_prefix}_M_vs_beta.png", dpi=160, bbox_inches='tight')


def demo():
    """Example: sweep β for a few σ values and plot bifurcation curves."""
    cfg = SolverConfig(L=20.0, N=1200, newton_tol=1e-10, newton_maxit=50, verbose=False)

    # Choose β sweep (start small to large); ensure domain is big enough so φ decays well
    beta_list = np.linspace(0.05, 1.0, 16)

    sigmas = [0.5, 1.0, 2.0]  # examples: subcubic, cubic, quintic-like power-law indices

    results_by_sigma = {}
    for sigma in sigmas:
        res = branch_over_beta(beta_list.tolist(), sigma, cfg, warm_start=True)
        results_by_sigma[sigma] = res

    plot_bifurcation(results_by_sigma, save_prefix="bifurcation")

    # Also show an example profile for reference (take last σ, last β)
    example = results_by_sigma[sigmas[-1]][-1]
    x, phi = example["x"], example["phi"]
    plt.figure()
    plt.plot(x, phi)
    plt.xlabel("x")
    plt.ylabel(r"φ(x)")
    plt.title(fr"Example profile: σ={example['sigma']}, β={example['beta']:.3f}, A={example['amplitude']:.3f}")
    plt.grid(True, alpha=0.3)
    plt.savefig("example_profile.png", dpi=160, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    demo()
