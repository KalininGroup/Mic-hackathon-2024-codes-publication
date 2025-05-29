# Team 10: Automating AFM through model-driven image segmentation and classification

## Team members

- Sam Welborn (NERSC/NCEM): team leader
- Mikolaj Jakowski
- Shawn Patrick
- Alex Pattison (NCEM)
- Panos Manganaris
- Sirisha Madugula

## Goals

Our goals were as follows:

1. Create an automated AFM workflow to discover areas of interest and feed this back into the microscope.
2. Use AI-driven image segmentation to divide the sample, and find areas of interest by classifying the image segments.
3. Enable reproducibility of this workflow via containerization.

## Progress

1. Miko and Shawn found that applying the [CellSAM](https://github.com/vanvalenlab/cellSAM) foundation model, typically used for cellular targets (bacteria, tissue, yeast, cell culture, etc.), worked very well to segment perovskite AFM data.
2. Alex used jupyter notebooks as scratch paper (see [this notebook](Cellsem_segmentation.ipynb)). Using CellSAM, he segmented AFM images and classified them by both size and number of nearest neighbors.
3. Sam created reproducible `Operators` for each of the functions involved in this workflow. These operators are deployed in the [interactEM](https://github.com/NERSC/interactEM) framework:

    - **[Microscope](operators/scope/)**: this operator brings up the DTMicroscope server and publishes a particular image for downstream segmentation.

    - **[Numpy array to image converter](operators/image_converter/)**: converts a numpy image array into an image. This is eventually published to interactEM's frontend via a websocket.

    - **[Image segmentation](operators/segmentation)**: takes in a numpy array, and segments it using [CellSAM](https://github.com/vanvalenlab/cellSAM). Outputs the mask.

    - **[Mask analysis](operators/post-process)**: takes in the mask, and outputs a list of coordinates. This was enabled by [#6](https://github.com/swelborn/team-10/pull/6) and [#7](https://github.com/swelborn/team-10/pull/7)

    - **[Microscope movement](operators/scope-downstream)**: this operator takes in the list of coordinates and "moves" to a particular location in the sample and scans around it. This is a fake operator for the time being as we didn't have time to implement a continuous feedback loop. In production, the output of the mask analysis operator would go back to the original microscope operator.

    Note that there are some additional `README`s inside of [operators](operators/) for more information.

4. Panos developed and understood DTMicroscope to collect scans in a particular range.

After building all of these containers and launching them with `interactEM`, we have the following:

![MIC2024-SLIDES](https://github.com/user-attachments/assets/b8d009a1-3e2c-45f1-a31e-c24ebfaa5a8a)

The data is loaded in the microscope operator, is passed to the segmentation operator, and downstream to the post-processing which selects a large grain. The final operator shows where the microscope should scan next.

## Future work

For the purposes of this demo, we use the output of the `Mask Analysis` (the list of coordinates) into another microscope operator (separate from the original) and went to locations on the "same" sample (the same original image). In order to fully close the loop, the original microcope operator should be notified of the locations to move to. This is possible with the `interactEM` platform, but not currently easily exposed.

We made several other more advanced data analysis functions. See [#10](https://github.com/swelborn/team-10/pull/10), [#11](https://github.com/swelborn/team-10/pull/11), [#12](https://github.com/swelborn/team-10/pull/12) and the functions in [extended_functions](extended_functions/). These functions could also be turned into operators to notify the microscope, however we did not have time to integrate them.

## Lessons learned

Image segmentation:

1. We hit a roadblock with image segmentation... our incoming images were not normalized. After normalizing our data, the segmentation model behaved properly.

Software development:

1. Many members of our team were unfamiliar with team software development/use of github. They were introduced to this concept through pull request/review and are now experts, see: [#6](https://github.com/swelborn/team-10/pull/6) and [#7](https://github.com/swelborn/team-10/pull/7).
2. We discussed the importance of type annotations in shared code. One can see the impact of this in the extended functions that Miko and Shawn made ([#10](https://github.com/swelborn/team-10/pull/10), [#11](https://github.com/swelborn/team-10/pull/11), [#12](https://github.com/swelborn/team-10/pull/12)).

## Acknowledgments

Thank you to the organizers for putting on a great hackathon! We enjoyed our time together. It was great to make new connections and explore new ways to integrate technology. Further, thanks to Rama for helping us convert `ibw` data into a format readable by `DTMicroscope`.
