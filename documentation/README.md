# MAS-LSTM
paper "MAS-LSTM: A Multi-Agent LSTM-Based Approach for Scalable Anomaly Detection in IIoT Networks" code.
 
# Datasets

The source for the full datasets made up of 43 extended NetFlow features is:
https://staff.itee.uq.edu.au/marius/NIDS_datasets/

To reproduce this work, download the 4 datasets (NF-ToN-IoT NF-BoT-IoT NF-ToN-IoT-v2 NF-BoT-IoT-v2) and compress them with Gzip, because the scripts reads files with the extension `*.csv.gz`.

## Requirements
- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- NumPy
