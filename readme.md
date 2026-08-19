# Shrinkage thesis codebase

#### Overview
Code should only be run from the notebooks. Before running any notebooks ensure that the LANL authentication events and red team events datasets are downloaded

##### Data download
The LANL dataset can be accessed at: https://csr.lanl.gov/data/cyber1/
The authentication data should be stored in data/raw as a parquet file
The redteam data should be stored in data/raw_redteam as a txt file

#### Running the codebase
To run the codebase run notebooks 100-204 in numberical order. The results will materialise in the outputs folder.
For test runs first set hyperparameter choices in the best configs json5 file
For validation runs set the hyperparameter choices you want to use in hyper_choices.json5 file

#### Notebook purposes
100 - initial preprocessing and train test split
200 - pre run processing
201 - validation tuning and test runs
202 - benchmark runs
203 - results analysis and plots
204 - variant runs for the appendix table in thesis. Not needed for core results