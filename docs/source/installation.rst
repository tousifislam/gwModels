Installation
============

From PyPI
---------

.. code-block:: bash

   pip install gwModels

From source
-----------

.. code-block:: bash

   git clone https://github.com/tousifislam/gwModels
   cd gwModels
   pip install -e .

After cloning from source, verify all model data files are present:

.. code-block:: bash

   python gwmodels_setup_data.py

Dependencies
------------

The following packages are required and will be installed automatically:

* `numpy <https://numpy.org/>`_
* `scipy <https://scipy.org/>`_
* `matplotlib <https://matplotlib.org/>`_
* `gwtools <https://pypi.org/project/gwtools/>`_

Optional dependencies for specific models:

* ``scikit-learn`` — for ``gwModel_kick_q200_GPR``
* ``torch``, ``nflows`` — for ``gwModel_kick_prec_flow``
* ``gwsurrogate`` — for NRHybSur3dq8-gwNRHME and BHPTNRSur1dq1e4-gwNRHME

Building the documentation
--------------------------

To build the documentation locally:

.. code-block:: bash

   pip install sphinx sphinx_rtd_theme nbsphinx
   cd docs
   make html

The built documentation will be in ``docs/build/html/``.
