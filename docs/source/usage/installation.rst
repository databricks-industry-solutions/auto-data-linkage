Package installation
####################

Databricks Runtime Requirements
**********************
ARC requires Databricks ML Runtime 12.2LTS. ARC will not work on 13.x or greater runtimes.

Installation from PyPI
**********************
Python users can install the library directly from `PyPI <https://pypi.org/project/databricks-mosaic/>`__
using the instructions `here <https://docs.databricks.com/libraries/cluster-libraries.html>`__
or from within a Databricks notebook using the :code:`%pip` magic command, e.g.

.. code-block:: bash

    %pip install databricks-arc

