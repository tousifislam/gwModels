Quick Start
===========

This guide shows the basic usage of ``gwModels`` for generating eccentric
waveforms, estimating eccentricity, and computing remnant properties.

Eccentric waveform with higher modes (gwNRHME)
----------------------------------------------

.. code-block:: python

   import gwModels

   # Generate eccentric higher-mode waveform using NRHybSur3dq8 + SEOBNRv5EHM
   t, h = gwModels.waveforms.NRHybSur3dq8_gwNRHME(
       q=4, s1z=0.0, s2z=0.0, ecc=0.1,
       mean_ano=0.0, f_low=20.0,
       delta_t=1.0/4096, M_tot=60.0,
   )

Eccentricity estimation
-----------------------

.. code-block:: python

   import gwModels

   ecc_calc = gwModels.ecc_measures.ComputeEccentricity(t, h22)
   e_xi = ecc_calc.compute_ecc_xi()

Eccentricity evolution models
-----------------------------

.. code-block:: python

   import gwModels

   # Fast approximate model
   model = gwModels.dynamics.gwEccEvNS()
   e_evolved = model.predict(q=2, e0=0.1, x0=0.05)

   # SVD surrogate model
   model = gwModels.dynamics.gwEccEvolve_NoSpinq4('data/gwEccEvolve_NoSpinq4.npy')
   result = model(q=2, e0=0.1)

Aligned-spin kick velocity
---------------------------

.. code-block:: python

   from gwModels.remnants import gwModel_kick_q200

   vk = gwModel_kick_q200(q=5, s1z=0.5, s2z=-0.3)
   print(f"Kick: {vk:.1f} km/s")

   # With uncertainty
   vk, vk_std = gwModel_kick_q200(q=5, s1z=0.5, s2z=-0.3, return_std=True)

Precessing kick distribution
-----------------------------

.. code-block:: python

   from gwModels.remnants import gwModel_kick_prec_flow

   model = gwModel_kick_prec_flow('data/')
   samples = model.sample(q=3, a1=0.5, a2=0.4, num_samples=5000)

Final mass and spin
-------------------

.. code-block:: python

   from gwModels.remnants import (
       bbh_final_mass_non_precessing_UIB2016,
       bbh_final_spin_non_precessing_UIB2016,
   )

   Mf = bbh_final_mass_non_precessing_UIB2016(m1=30, m2=20, chi1=0.5, chi2=-0.3)
   chif = bbh_final_spin_non_precessing_UIB2016(m1=30, m2=20, chi1=0.5, chi2=-0.3)
