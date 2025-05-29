# Microscope Operator

This operator initiates a microscope server and serves the same image every 60 seconds. In production, we would provide the real microscope with directions on where to go from the downstream operators.

The data is actually mounted in at runtime via interactEM as a parameter. We could also "build in" the dataset to the image, but we wouldn't do this in production. We have provided the dataset we used under `assets` in this directory.

This operator is connected to an image segmentation operator. It wraps metadata into a json-serialized string as a header, and data as bytes.
