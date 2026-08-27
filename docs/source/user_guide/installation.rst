============
Installation
============

This tutorial guides you through installing **seq-control** on your local machine, whether you want to use it as a dependency in another project or contribute to its development.

Prerequisites
-------------

Before installing, ensure you have the following installed on your system:

* Python 3.8 or higher
* ``git`` command line tool
* ``pip`` (Python package manager)

Option 1: Quick Install via Git & pip
--------------------------------------

If you simply want to use ``seq-control`` in your Python scripts without modifying the source code, you can install it directly from GitHub using ``pip``.

1. Open your terminal or command prompt.
2. Run the following command:

.. code-block:: bash

    pip install git+https://github.com/ilyes0702/seq-control.git

3. To install a specific version or release tag in the future, append ``@tag_name`` to the URL:

.. code-block:: bash

    pip install git+https://github.com/ilyes0702/seq-control.git@v1.0.0

Option 2: Developer / Local Installation
---------------------------------------

If you want to modify the source code, build new features, or run tests, set up an editable local installation.

Step 1: Clone the Repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, download a local copy of the repository using ``git``:

.. code-block:: bash

    git clone https://github.com/ilyes0702/seq-control.git
    cd IAA_Master_Thesis_DPE

Step 2: Install in Editable Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install the package using the ``-e`` flag. This links your environment to the source files in ``src/seq-control`` so that any changes you make to the code take effect immediately without re-installing:

.. code-block:: bash

    pip install -e .

.. tip::
   If you plan to run the test suite or build documentation locally, install the developer dependencies as well:

   .. code-block:: bash

       pip install -r requirements.txt

Verifying Your Installation
---------------------------

To verify that ``seq-control`` is correctly installed and accessible in your Python environment:

1. Launch a Python interactive shell:

.. code-block:: bash

    python

2. Try importing the module:

.. code-block:: python

    import seq-control
    print("seq-control successfully installed!")
