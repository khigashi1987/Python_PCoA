Python_PCoA
===========

The Python implementation for Principal Coordinate Analysis.
For distance metric, one of Jaccard, Bray-Curtis, or Jensen-Shannon divergence can be used.


    usage: pcoa.py [-h] [-f DATA_FILE] [-d {Jaccard,BrayCurtis,JSD}] [-b]
                   [-n N_ARROWS] [-g GROUP_FILE]
    
    optional arguments:
      -h, --help            show this help message and exit
      -f DATA_FILE, --file DATA_FILE
                            matrix data file. rows are variables, columns are
                            samples.
      -d {Jaccard,BrayCurtis,JSD}, --distance_metric {Jaccard,BrayCurtis,JSD}
                            choose distance metric used for PCoA.
      -b, --biplot          output biplot (with calculating factor loadings).
      -n N_ARROWS, --number_of_arrows N_ARROWS
                            how many top-contributing arrows should be drawed.
      -g GROUP_FILE, --grouping_file GROUP_FILE
                            plot samples by same colors and markers when they
                            belong to the same group. Please indicate Tab-
                            separated 'Samples vs. Group file' ( first columns are
                            sample names, second columns are group names ).
