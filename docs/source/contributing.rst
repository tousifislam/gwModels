Contributing
============

Contributions to ``gwModels`` are welcome! Here's how you can help:

Reporting Issues
----------------

If you find a bug or have a feature request, please open an issue on the
`GitHub issue tracker <https://github.com/tousifislam/gwModels/issues>`_.

Development Setup
-----------------

1. Fork and clone the repository:

   .. code-block:: bash

      git clone https://github.com/your-username/gwModels.git
      cd gwModels

2. Install in development mode:

   .. code-block:: bash

      pip install -e .

3. Make your changes and ensure the package still imports correctly:

   .. code-block:: bash

      python -c "import gwModels; print('OK')"

4. If you modified documentation, build and check it:

   .. code-block:: bash

      cd docs
      make html

Pull Requests
-------------

1. Create a new branch for your feature or fix.
2. Make your changes with clear commit messages.
3. Submit a pull request with a description of what you changed and why.

Code Style
----------

* Follow PEP 8 conventions.
* Use NumPy-style docstrings for functions and classes.

Contact
-------

For questions, suggestions, or collaborations, please email tousifislam24@gmail.com.
