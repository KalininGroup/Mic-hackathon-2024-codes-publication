# Post-processing operator

This operator takes the mask, and performs some analysis on it. There are two main functions here:

1. `select_samples_grain_size` -- this will select either the largest or smallest grains in the sample and send the coordinates upstream.

2. `find_high_boundary_areas` -- this will get the grains with the highest number of grain boundaries and send the coordinates upstream.

The "parameters" here are controllable through the frontend. For example, I can choose between the `select_samples_for_grain_size` or `find_high_boundary_areas` in the frontend. Further, I can choose the number of bins and samples to use in the `select_samples_for_grain_size`, and the `num_targets` parameter for the `find_high_boundary_areas` function. This is controlled through the interactEM web interface by selecting the cog on the operator.
