This is the data and program for the unpublished paper 'Bargaining-based Allocation of Dynamic Operating Envelopes in Distribution Networks'.

## Data
+ The network parameters and aggregators' DER data are in `141_system.xls`. In the main program, we modify a little bit on the DER data such as the PV generation.
+ Other `.npy` and `.mat` files are supplementart data for the simultion.

## Program
+ `DOE allocation 141case single_interval.py` is the main program for the case study in the paper.
+ `utilis4biddingRW_single_interval_normalizedNB.py` contains the objects that are useful for building the network, generating prosumers and aggregators, and allocating DOEs.
