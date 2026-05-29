gwModels
========

.. image:: ../../gwModels.png
   :align: center
   :width: 400px

**gwModels** is a Python package hosting data-driven and phenomenological
models for the gravitational radiation (waveforms) emitted from binary
black hole mergers, along with remnant property models for final mass,
spin, and kick velocity.

For questions, suggestions, or collaborations, please email tousifislam24@gmail.com.

Available Models
----------------

**Waveform Frameworks**

.. list-table::
   :header-rows: 1
   :widths: 25 50 15 10

   * - Model
     - Description
     - Reference
     - Tutorial
   * - **gwNRHME**
     - Non-spinning quasi-circular HM waveform → eccentric
     - `2403.15506 <https://arxiv.org/abs/2403.15506>`_
     - `1_1 <notebooks/1_1_framework_gwNRHME_example.html>`_
   * - **gwNRXHME**
     - Non-precessing quasi-circular HM waveform → eccentric
     - `2502.02739 <https://arxiv.org/abs/2502.02739>`_
     - `1_1 <notebooks/1_1_framework_gwNRHME_example.html>`_

**Eccentric Higher-Mode Waveforms**

.. list-table::
   :header-rows: 1
   :widths: 30 30 15 10

   * - Model
     - Components
     - Reference
     - Tutorial
   * - **NRHybSur3dq8-gwNRHME**
     - NRHybSur3dq8 + SEOBNRv5EHM
     - `2408.02762 <https://arxiv.org/abs/2408.02762>`_
     - `2_1 <notebooks/2_1_NRHybSur3dq8-gwNRHME_example.html>`_
   * - **BHPTNRSur1dq1e4-gwNRHME**
     - BHPTNRSur1dq1e4 + SEOBNRv5EHM
     - `2408.02762 <https://arxiv.org/abs/2408.02762>`_
     - `2_2 <notebooks/2_2_BHPTNRSur1dq1e4-gwNRHME_example.html>`_

**Eccentricity Estimation**

.. list-table::
   :header-rows: 1
   :widths: 30 40 15 10

   * - Model
     - Description
     - Reference
     - Tutorial
   * - **eccentricity_estimation**
     - Computes :math:`e_{\xi}`, :math:`e_{\omega}`, :math:`e_{\rm gw}`
     - `2502.02739 <https://arxiv.org/abs/2502.02739>`_
     - `3_1 <notebooks/3_1_eccentricity_estimation_nonprecessing.html>`_

**Dynamics: Eccentricity Evolution**

.. list-table::
   :header-rows: 1
   :widths: 25 20 25 15 10

   * - Model
     - Type
     - Parameter Range
     - Reference
     - Tutorial
   * - **gwEccEvNS**
     - NR-based approximate
     - Non-spinning
     - `2502.02739 <https://arxiv.org/abs/2502.02739>`_
     - `4_1 <notebooks/4_1_dynamics_gwEccEvNS.html>`_
   * - **gwEccEvNSv2**
     - Analytical
     - Non-spinning
     - `2604.17868 <https://arxiv.org/abs/2604.17868>`_
     - `4_2 <notebooks/4_2_dynamics_gwEccEvNSv2.html>`_
   * - **gwEccEvolve_NoSpinq4**
     - SVD surrogate + GPR
     - :math:`1 \leq q \leq 4`, :math:`0.003 \leq e_0 \leq 0.443`
     - `2604.17868 <https://arxiv.org/abs/2604.17868>`_
     - `4_3 <notebooks/4_3_dynamics_gwEccEvolve_NoSpinq4.html>`_

**Remnant: Kick Velocity**

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 15 15 10

   * - Model
     - Type
     - Valid Range
     - Extra Deps
     - Reference
     - Tutorial
   * - **gwModel_kick_q200**
     - Analytical (aligned)
     - :math:`1 \leq q \leq 1000`
     - —
     - `2511.11536 <https://arxiv.org/abs/2511.11536>`_
     - `5_1 <notebooks/5_1_gwModels_kicks.html>`_
   * - **gwModel_kick_q200_GPR**
     - GPR (aligned)
     - :math:`1 \leq q \leq 1000`
     - scikit-learn
     - `2511.11536 <https://arxiv.org/abs/2511.11536>`_
     - `5_1 <notebooks/5_1_gwModels_kicks.html>`_
   * - **gwModel_kick_prec_flow**
     - Normalizing flow (prec.)
     - :math:`q \leq 100`
     - torch, nflows
     - `2511.11536 <https://arxiv.org/abs/2511.11536>`_
     - `5_1 <notebooks/5_1_gwModels_kicks.html>`_
   * - **HLZ_2014_aligned_spin**
     - Analytical (aligned)
     - —
     - —
     - `1406.7295 <https://arxiv.org/abs/1406.7295>`_
     - `5_2 <notebooks/5_2_other_remnant_models.html>`_
   * - **bbh_final_kick_precessing_CLZM2007**
     - Analytical (prec.)
     - —
     - —
     - Gonzalez+ 2007
     - `5_2 <notebooks/5_2_other_remnant_models.html>`_

**Remnant: Final Mass and Spin**

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 25 10

   * - Model
     - Quantity
     - Spin Type
     - Reference
     - Tutorial
   * - **bbh_final_mass_precessing_BMR2012**
     - Final mass
     - Precessing
     - Barausse, Morozova & Rezzolla (2012)
     - `5_2 <notebooks/5_2_other_remnant_models.html>`_
   * - **bbh_final_spin_precessing_HBR2016**
     - Final spin
     - Precessing
     - Hofmann, Barausse & Rezzolla (2016)
     - `5_2 <notebooks/5_2_other_remnant_models.html>`_
   * - **bbh_final_mass_non_precessing_UIB2016**
     - Final mass
     - Aligned
     - `1611.00332 <https://arxiv.org/abs/1611.00332>`_
     - `5_2 <notebooks/5_2_other_remnant_models.html>`_
   * - **bbh_final_spin_non_precessing_UIB2016**
     - Final spin
     - Aligned
     - `1611.00332 <https://arxiv.org/abs/1611.00332>`_
     - `5_2 <notebooks/5_2_other_remnant_models.html>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   api/index
   notebooks/index
   citation
   contributing
