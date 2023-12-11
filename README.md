"# pollutantforecastingthesis23" 

During the writing of this project, Google Colab updated its Ubuntu (20.04 LTS to 22.04 LTS) and Cuda (to 12.0) drivers. This took place between the first no-neighbors (n=0) training model variant 1, and the with-neighbors (n=3) training model variants 1 and 2.

The most significant impact was the version of MXNet used had to be updated.
- Before the Google Colab update, a downgrade of the Cuda drivers was performed by the notebook files, to use `mxnet-cu110`.
- After the Google Colab update, the version of MXNet had to become the latest `mxnet-cu112` to be able to run; downgrading Cuda to 11.0 no longer worked because Ubuntu was updated.

Also, the MXNet project became officially archived by Apache, or abandoned.


## Significant files used _BEFORE_ the update:

### code:

- **data_filtering.py, city_neighbor.py** -- prepared and sequenced data for training with no neighbors (n=0).
- **mxnet-together.ipynb** -- experimented with training variant 1 with no neighbors (n=0).
- **testing-gpu-mxnet-again.ipynb** -- experimented with training variant 1 on GPU.

### cached data:

- **city_padded_seqs_noneighbors/\*** -- cached sequence data intended for no-neighbors training
- **city_seqs_prejoined/\*** -- cached sequence data intended for neighbors training (was superseded by later training)
- **city_seq_prejoined_dict.pickle** -- cached sequence data intended for neighbors training (was superseded by later training)

- **filtered_city_neighbors_cache.json, filtered_jingjinji.csv, filtered_pm25_stdized_info.json** -- these files were cached data created and used for the old (n=0) training, imported in code files such as mxnet-together.ipynb.

- **net.params** -- saved trained network parameters used in an example in mxnet-together.ipynb to test parameter saving.
- **params/\*** -- saved trained network parameters and others of variant 1 and experimental. Created and used by mxnet-together.ipynb and testing-gpu-mxnet-again.ipynb.


## Significant files used _AFTER_ the update:

### code:

- **mxnet_prepare_sequences.ipynb** -- creates the sequenced data from the raw CSVs. The sequenced data are the tensors referred to in the paper as _T_ and _N_ and _y_, that will be fed into the models for fitting and used for error metrics.
  Saves the sequenced data to the /prepared_seqs/ folder.
- **data_utils_df_normalizer.py, data_utils_df_joiner.py** -- used by mxnet_prepare_sequences.ipynb to process the data.
- **mxnet_train_from_seqs.ipynb** -- defines and creates NEP-LSTM variants 1 and 2, and trains them on the sequenced data, from the /prepared_seqs/ folder.
  Saves the network parameters to the /params2/ folder for future loading of the trained networks. Also saves the losses to the /params2/ folder.
- **train_sklearn_random_forest_tree.ipynb** -- creates the Random Forest trees, sklearn RandomForestRegressors, and trains them on the sequenced data, from the /prepared_seqs/ folder.
  Flattening is performed (using `numpy`'s `.ravel()`) to provide sequences as arrays to the Random Forest trees.
- **graph-losses.ipynb** -- loads the losses saved to the /params2/ folder, as well as the losses from the original variant 1 n=0 training, and graphs them.
  Also calculates and graphs the feature importances and the Anscombe's quartet.

### cached data:

- **prepared_seqs/\*** -- cached sequence data used for later neighbors training (n=3)
- **params2/\*** -- saved trained network parameters and others of variant 1 and 2, (n=3).
- **presequencing_means_stds_dict.json** -- created by mxnet_prepare_sequences.ipynb, and used by mxnet_prepare_sequences.ipynb and mxnet_train_from_seqs.ipynb. Contains the 'original' means and standard deviations for the pollutant features that were standardized during processing.