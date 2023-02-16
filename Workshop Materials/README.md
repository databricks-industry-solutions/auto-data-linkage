# splink-public-sector-hackathon
A hackathon for Entity Resolution using Splink (by MoJ) on Azure Databricks done by NICD, MoJ, Microsoft and Databricks


# How to Run

1. Add repository to your Databricks Workspace via Repos. Paste this link when required: https://github.com/databricks-industry-solutions/splink-public-sector-hackathon.git

2. Run the *Source data from Kaggle and Companies House* notebook to create the input data. NB - this takes about 7 hours to complete. This notebook should be executed on a cluster running DBR 11.3 LTS.

3. Walk through the *end-to-end example* notebook in the `notebooks` directory to understand how Splink works. This notebook should be executed on a cluster running DBR 12.1 ML

4. Use the *exercise* notebook as a starting point for your own work. Consider exploring different parameters or feature engineering to understand the impact on linking. Visualise the results using Python, R, or the built in visualisation tools in Databricks Notebooks or Databricks SQL.

5. Look at the *Splink in Databricks - Expert Answer* notebook to see how an expert user of Splink might approach this problem. 

For more information, please reach out to robert.whiffin@databricks and Sam.Lindsay1@justice.gov.uk.