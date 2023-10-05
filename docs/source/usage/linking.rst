Data Linking with ARC
####################

Getting started
**********************

Starting a linking project with ARC is simple - you just need to load your data into a pair of Spark Dataframes, and pass
them to the Autolinker object in a list.

.. code-block:: python
    from arc.autolinker import AutoLinker

    autolinker = AutoLinker()

    data_1 = spark.read.load(<a table of messy data>)
    data_2 = spark.read.load(<a table of messy data>)
    data = list(data_1, data_2)
    autolinker.auto_link(
      data=data
    )

No other options are required. This will start an auto linking process that will run for 5 iterations of hyperparameter
optimisation. During these 5 rounds ARC will try a variety of options to determine impact on model performance. It is
unlikely that 5 rounds will be sufficient - in our testing we found that the model needed to run for 100 iterations or
more for good results.

This is done with

.. code-block:: python
    from arc.autolinker import AutoLinker

    autolinker = AutoLinker()

    data_1 = spark.read.load(<a table of messy data>)
    data_2 = spark.read.load(<a table of messy data>)
    data = list(data_1, data_2)
    autolinker.auto_link(
      data=data
      ,max_evals=100
    )
