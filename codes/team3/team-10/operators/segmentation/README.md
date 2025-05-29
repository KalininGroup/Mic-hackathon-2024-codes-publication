# Segmentation operator

This ingests data from the microscope and segments the image. Despite attempting running using the `mps` backend, this is not currently a supported feature of pytorch when running inside of a container. In production, this operator would run on a GPU-capable machine, making the segmentation significantly faster.

This exports a mask to the downstream operators.