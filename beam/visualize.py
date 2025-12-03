"""
Visualization Tools for BEAM

Provides plotting functions for:
- TICA projections
- Free energy landscapes
- CG/AA overlay visualization (key figure!)
- Residue contributions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_tica_projection(cv_data, title="TICA Projection", 
                         xlabel="tIC 1", ylabel="tIC 2",
                         color_by_time=True, save_path=None,
                         figsize=(7, 6)):
    """
    Plot 2D TICA projection.
    
    Parameters
    ----------
    cv_data : np.ndarray
        CV trajectory, shape (n_frames, n_cv) where n_cv >= 2
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    color_by_time : bool
        If True, color points by frame index (time progression)
    save_path : str, optional
        Save figure to this path
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if color_by_time:
        scatter = ax.scatter(
            cv_data[:, 0], cv_data[:, 1],
            c=np.arange(len(cv_data)),
            cmap='viridis',
            s=3,
            alpha=0.6,
            rasterized=True
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Frame Index', fontsize=10)
    else:
        ax.scatter(
            cv_data[:, 0], cv_data[:, 1],
            c='steelblue',
            s=3,
            alpha=0.6,
            rasterized=True
        )
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()


def plot_free_energy_landscape(cv_data, bins=50, 
                               title="Free Energy Landscape",
                               xlabel="tIC 1", ylabel="tIC 2",
                               vmax=None, save_path=None,
                               figsize=(8, 7)):
    """
    Plot 2D free energy landscape from CV trajectory.
    
    Parameters
    ----------
    cv_data : np.ndarray
        CV trajectory, shape (n_frames, n_cv) where n_cv >= 2
    bins : int or [int, int]
        Number of bins for 2D histogram
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    vmax : float, optional
        Maximum value for colormap (in kT units)
    save_path : str, optional
        Save figure to this path
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute 2D histogram
    H, xedges, yedges = np.histogram2d(
        cv_data[:, 0], cv_data[:, 1],
        bins=bins
    )
    
    # Convert to free energy (kT units)
    H = H + 1e-10  # Avoid log(0)
    F = -np.log(H)
    F = F - np.min(F)  # Set minimum to 0
    
    # Plot
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    im = ax.imshow(
        F.T,
        origin='lower',
        extent=extent,
        cmap='coolwarm',
        aspect='auto',
        vmax=vmax,
        interpolation='bilinear'
    )
    
    # Add contour lines
    contour = ax.contour(
        F.T,
        extent=extent,
        colors='black',
        alpha=0.3,
        linewidths=0.5,
        levels=10
    )
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Free Energy (kT)', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()


def plot_cg_aa_overlay(cg_cv, aa_cv, 
                       title="AA Enhanced Sampling on CG-learned CV Space",
                       xlabel="CG tIC 1", ylabel="CG tIC 2",
                       cg_color='gray', cg_alpha=0.2, cg_size=1,
                       aa_cmap='plasma', aa_size=8, aa_alpha=0.8,
                       save_path=None, figsize=(9, 8)):
    """
    Overlay AA trajectory on CG training data in CV space.
    
    **This is the key visualization for BEAM!**
    
    Shows:
    - CG data (gray background) = training space
    - AA data (colored by time) = enhanced sampling trajectory
    
    This figure demonstrates that:
    1. CG simulations explored conformational space
    2. AA enhanced sampling is guided by CG-learned CVs
    3. AA trajectory follows/extends CG predictions
    
    Parameters
    ----------
    cg_cv : np.ndarray
        CG trajectory in CV space, shape (n_frames_cg, n_cv)
    aa_cv : np.ndarray
        AA trajectory in CV space, shape (n_frames_aa, n_cv)
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    cg_color : str
        Color for CG data points
    cg_alpha : float
        Transparency for CG data
    cg_size : float
        Size of CG data points
    aa_cmap : str
        Colormap for AA data (colored by time)
    aa_size : float
        Size of AA data points
    aa_alpha : float
        Transparency for AA data
    save_path : str, optional
        Save figure to this path
    figsize : tuple
        Figure size
        
    Examples
    --------
    >>> # Load CG and AA CV trajectories
    >>> cg_cv = np.load('cg_cv_trajectory.npy')
    >>> aa_cv = np.load('aa_in_cg_cv_space.npy')
    >>> 
    >>> # Create overlay plot
    >>> plot_cg_aa_overlay(cg_cv, aa_cv, save_path='overlay.png')
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot CG data (gray background)
    ax.scatter(
        cg_cv[:, 0], cg_cv[:, 1],
        c=cg_color,
        alpha=cg_alpha,
        s=cg_size,
        label='CG training data',
        rasterized=True
    )
    
    # Plot AA data (colored by time)
    scatter = ax.scatter(
        aa_cv[:, 0], aa_cv[:, 1],
        c=np.arange(len(aa_cv)),
        cmap=aa_cmap,
        alpha=aa_alpha,
        s=aa_size,
        label='AA enhanced sampling',
        edgecolors='none',
        rasterized=True
    )
    
    # Colorbar for AA trajectory time
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('AA Frame Index (Time)', fontsize=11)
    
    # Labels and title
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, pad=10)
    
    # Legend
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    
    # Grid
    ax.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()
    
    print("\nFigure interpretation:")
    print("  • Gray points: CG training data (explored conformational space)")
    print("  • Colored points: AA trajectory (guided by CG-learned CVs)")
    print("  • Color gradient: Time evolution of AA sampling")
    print("\nThis demonstrates the BEAM workflow:")
    print("  1. CG simulations discover slow modes")
    print("  2. CVs guide AA enhanced sampling")
    print("  3. AA trajectory validates/extends CG predictions")


def plot_residue_contributions(residue_weights, 
                               residue_range=None,
                               title="Residue Contributions to CV",
                               xlabel="Residue Index",
                               ylabel="Contribution Weight",
                               color='steelblue',
                               save_path=None,
                               figsize=(12, 4)):
    """
    Plot residue-level contributions to a collective variable.
    
    Shows which residues contribute most to the slow mode.
    Useful for mechanistic interpretation.
    
    Parameters
    ----------
    residue_weights : np.ndarray
        Contribution weight per residue
    residue_range : tuple, optional
        (start, end) residue numbers for x-axis
        If None, uses 1, 2, 3, ...
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    color : str
        Bar color
    save_path : str, optional
        Save figure to this path
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if residue_range is None:
        residue_indices = np.arange(1, len(residue_weights) + 1)
    else:
        residue_indices = np.arange(residue_range[0], residue_range[1] + 1)
    
    ax.bar(
        residue_indices,
        residue_weights,
        color=color,
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Highlight top contributors
    top_indices = np.argsort(np.abs(residue_weights))[-5:]
    for idx in top_indices:
        ax.axvline(residue_indices[idx], color='red', 
                   alpha=0.3, linestyle='--', linewidth=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()


def plot_cv_comparison(cg_cv, aa_in_cg_cv, aa_in_aa_cv=None,
                      save_path=None, figsize=(15, 5)):
    """
    Compare CV distributions across CG and AA.
    
    Shows:
    - CG CV distribution (training data)
    - AA in CG CV space (how AA samples the CG-learned space)
    - (Optional) AA in AA CV space (AA-specific slow modes)
    
    Parameters
    ----------
    cg_cv : np.ndarray
        CG trajectory in CG CV space
    aa_in_cg_cv : np.ndarray
        AA trajectory in CG CV space
    aa_in_aa_cv : np.ndarray, optional
        AA trajectory in AA CV space
    save_path : str, optional
        Save figure to this path
    figsize : tuple
        Figure size
    """
    n_plots = 2 if aa_in_aa_cv is None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    # Plot 1: CG CV
    axes[0].scatter(cg_cv[:, 0], cg_cv[:, 1], 
                    c='gray', alpha=0.3, s=1, rasterized=True)
    axes[0].set_title('CG in CG CV Space', fontsize=12)
    axes[0].set_xlabel('CG tIC 1')
    axes[0].set_ylabel('CG tIC 2')
    
    # Plot 2: AA in CG CV
    scatter1 = axes[1].scatter(aa_in_cg_cv[:, 0], aa_in_cg_cv[:, 1],
                               c=np.arange(len(aa_in_cg_cv)),
                               cmap='plasma', s=5, rasterized=True)
    axes[1].set_title('AA in CG CV Space', fontsize=12)
    axes[1].set_xlabel('CG tIC 1')
    axes[1].set_ylabel('CG tIC 2')
    plt.colorbar(scatter1, ax=axes[1], label='Frame')
    
    # Plot 3: AA in AA CV (if provided)
    if aa_in_aa_cv is not None:
        scatter2 = axes[2].scatter(aa_in_aa_cv[:, 0], aa_in_aa_cv[:, 1],
                                   c=np.arange(len(aa_in_aa_cv)),
                                   cmap='viridis', s=5, rasterized=True)
        axes[2].set_title('AA in AA CV Space', fontsize=12)
        axes[2].set_xlabel('AA tIC 1')
        axes[2].set_ylabel('AA tIC 2')
        plt.colorbar(scatter2, ax=axes[2], label='Frame')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    plt.show()

def plot_timescales(tica_estimator, lagtime, dt=1.0, n_components=5, 
                   title="TICA Implied Timescales", save_path=None):
    """
    Plot implied timescales from TICA eigenvalues.
    
    Parameters
    ----------
    tica_estimator : deeptime.decomposition.TICA
        Fitted TICA estimator (not model)
    lagtime : int
        Lag time used for TICA (in frames)
    dt : float
        Time per frame (e.g., in ns)
    n_components : int
        Number of components to plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 获取 eigenvalues
    if hasattr(tica_estimator, 'eigenvalues'):
        eigvals = tica_estimator.eigenvalues[:n_components]
    else:
        print("Warning: Cannot access eigenvalues from this object")
        return
    
    # 计算 implied timescales
    tau = lagtime * dt
    timescales = -tau / np.log(np.abs(eigvals))
    
    # 画图
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(1, len(timescales) + 1), timescales, alpha=0.7)
    ax.set_xlabel('TICA Component', fontsize=12)
    ax.set_ylabel(f'Implied Timescale ({dt} per frame units)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(1, len(timescales) + 1))
    ax.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    
    return timescale
