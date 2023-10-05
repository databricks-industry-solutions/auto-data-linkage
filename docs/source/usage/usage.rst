.. toctree::
   :maxdepth: 2

   installation
   best_practices

Data Linking and Deduplication with ARC
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


However, ARC has many other optional arguments which can be used to fine tune its behaviour.  These are detailed below,
along with an explanation of when you would use them.


`data: typing.Union[pyspark.sql.dataframe.DataFrame, list]`

The input data to ARC. A single Dataframe will perform deduplication, a list of 2 dataframes will perform linking between
the two datasets. If linking, for best performance ensure schemas are standardised prior to linking (i.e. the 2 input
tables should share an identical schema). This is not required - ARC can handle mismatching schemas, but you will get
better performance with identical schemas.

`attribute_columns:list=None`

Which columns should be used for evaluating if records are connected. In the case of linking 2 datasets, will only work with
a identical schemas. If not provided, ARC will use all columns (except a unique ID column if provided) as input attributes
to the model.

**When to use**

- if you have very wide tables with lot of extraneous information which will not help the model in determining if records
  represent the same thing.
- if you want to link at a coarser granularity than the data are. For example, if you have a table of
  people and addresses and your aim is to link addresses, you could either drop the columns containing the people information, or use the *attribute_columns* arg to specify only the address information columns.

`unique_id:str=None`

Which column contains a unique per record ID. ARC will ignore this for purposes of linking. If None, ARC will append a new
column to the dataset called *unique_id*.

**When to use**
- if your data has a unique id column. ARC will evaluate this column for linking if not, which at best will not help produce a good model.

`comparison_size_limit:int=100000`

The maximum number of pairs of records a blocking rule will be allowed to have. Blocking rules are heuristics used by Splink
to control which records are potential duplicates. ARC auto-generates blocking rules for you, and uses this parameter to control
which blocking rules are used by looking at the amount of potential duplicates each rule generates.

**When to use**

- if you want to speed up the model training process, try setting a lower value, i.e. 50,000.
- this parameter will directly impact the recall of the model - if it is too low, you risk missing pairs.
  It is better to err on the side of too big than too small.
- it is generally unnecessary to change this value.

`max_evals:int=5`

The maximum number of evaluations ARC will do during its hyperparameter optimisation search. As a default value,
5 will give a taste of how ARC works, but internal testing showed good linking results from at least 100 runs.

**When to use**

- set to 100 or more when training a model for proper evaluation - this will take a long time, but it is reflective of
the size of search space across which HyperOpt needs to explore to find the best set of arguments.
- this should only be left as the default argument during initial testing and evaluation.

`cleaning="all"`

Provide an option of "all" or "none". ARC will lowercase and remove non-alphanumeric characters from all string columns if set to all.

**When to use**

- if you don't want ARC to do string cleaning for you

`threshold:float=0.9`

The probability threshold above which a pair is considered a match. This is used in the optimisation process.
Requires a value between 0 and 1.

**When to use**

- use this to balance precision and recall - higher number -> more emphasis on precision, less on recall.
  Lower value -> more emphasis on recall, less on precision.


`true_label:str=None`

The column name which contains the true record IDs. If provided, ARC will automatically score its model against the known true labels.

**When to use**

- if you have an already deduplicated / linked set of data that you want to use as a benchmark to assess ARC's performance.

`random_seed:int=42`
Random seed for controlling the psuedoRNG. Set for reproducibility between runs.

`metric:str="information_gain_power_ratio"`

Which metric should ARC use for it's optimisation process. This will be deprecated in future releases and should not be changed.


`sample_for_blocking_rules=True`

Downsample larger datasets to speed up blocking rule estimation.