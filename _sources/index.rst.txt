.. ARC documentation master file, created by
   sphinx-quickstart on Wed Feb  2 11:01:42 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ARC's documentation!

Documentation
=============

.. toctree::
   :glob:
   :titlesonly:
   :maxdepth: 2
   :caption: Contents:
   usage/usage


AutoLinker
----------
.. autoclass:: arc.autolinker.AutoLinker
    :members:
    :undoc-members:
    :private-members:


MLflow utilities
----------------

Wrapper classes
~~~~~~~~~~~~~~~
.. autoclass:: arc.autolinker.splink_mlflow.splinkSparkMLFlowWrapper
   :members:
   :undoc-members:
   :private-members:

.. autoclass:: arc.autolinker.splink_mlflow.SplinkMLFlowWrapper
   :members:
   :undoc-members:
   :private-members:


Utility functions
~~~~~~~~~~~~~~~~~
.. autofunction:: arc.autolinker.splink_mlflow._log_comparison_details

.. autofunction:: arc.autolinker.splink_mlflow._log_comparisons

.. autofunction:: arc.autolinker.splink_mlflow._log_hyperparameters

.. autofunction:: arc.autolinker.splink_mlflow._log_splink_model_json

.. autofunction:: arc.autolinker.splink_mlflow._save_splink_model_to_mlflow

.. autofunction:: arc.autolinker.splink_mlflow.log_splink_model_to_mlflow

.. autofunction:: arc.autolinker.splink_mlflow._log_linker_charts

.. autofunction:: arc.autolinker.splink_mlflow._log_chart

.. autofunction:: arc.autolinker.splink_mlflow.get_model_json



Indices and tables
==================

* :ref:`genindex`
* :ref:`search`


.. * :ref:`modindex`


Project Support
===============

Please note that all projects in the ``industry-solutions`` github space are provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support.
