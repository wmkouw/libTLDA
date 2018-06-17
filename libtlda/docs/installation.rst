************
Installation
************

libTLDA is registered on PyPI and can be installed through:

.. code-block:: bash

    pip install libtlda

Virtual environment
-------------------

Pip takes care of all dependencies, but the addition of these dependencies can mess up your current python environment. To ensure a clean install, it is recommended to set up a virtual environment using `conda <https://conda.io/docs/>`_ or `virtualenv <https://virtualenv.pypa.io/en/stable/>`_. To ease this set up, an environment file is provided, which can be run through:

.. code-block:: bash

    conda env create -f environment.yml
    source activate libtlda

For more information on getting started, see the Examples section.
