#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#     FILE: rcparams.py
#     Matplotlib plotting configuration for gwModels
#
#     AUTHOR: Tousif Islam
#     CREATED: 07-02-2024
#     REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"


def set_rcparams():
    """
    Apply gwModels matplotlib style settings.

    Call this explicitly to configure matplotlib for publication-quality plots.
    Requires a LaTeX installation for text.usetex.
    """
    import matplotlib.pyplot as plt

    plt.rc('figure', figsize=(8, 5))

    # Try to enable LaTeX rendering; fall back to mathtext if LaTeX is not installed
    try:
        plt.rcParams.update({
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath}',
        })
        # Force a render to check if LaTeX actually works
        import matplotlib
        matplotlib.texmanager.TexManager().get_grey("$x$")
    except Exception:
        import logging
        logging.getLogger(__name__).info("LaTeX not available, falling back to mathtext rendering")
        plt.rcParams.update({
            'text.usetex': False,
        })

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Georgia'],
        'mathtext.fontset': 'cm',
        'lines.linewidth': 1.2,
        'font.size': 14,
        'xtick.labelsize': 'medium',
        'ytick.labelsize': 'medium',
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 1.4,
        'ytick.major.width': 1.4,
        'xtick.major.size': 4.,
        'ytick.major.size': 4.,
        'ytick.right': True,
        'axes.labelsize': 'large',
        'axes.titlesize': 'large',
        'axes.grid': True,
        'grid.alpha': 0.5,
        'lines.markersize': 12,
        'legend.borderpad': 0.2,
        'legend.fancybox': True,
        'legend.fontsize': 12,
        'legend.framealpha': 0.7,
        'legend.handletextpad': 0.5,
        'legend.labelspacing': 0.2,
        'legend.loc': 'best',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.dpi': 80,
        'pdf.compression': 9,
    })
