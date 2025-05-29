<div align="center">
   
# CT4Batt: A pipeline for Analyzing X-ray Chromatography Battery Images
   
<img src="header_battery.png" alt="Header Image" style="width:75%;" />

</div>

This project explores a two-step machine learning approach for analyzing X-ray images of electrode stacks, using publicly available data from the X-ray-PBD dataset. The overarching goal is to identify and zoom in on the electrode tab region, and then isolate each individual electrode tab through segmentation.

In the first step, we focus on automatically locating a bounding box around the electrode tabs. We introduce a specialized convolutional neural network (CNN) equipped with attention mechanisms. This network employs coordinate attention, sliding window attention, and residual feature extractors to refine its understanding of the complex electrode structure. By predicting bounding boxes directly from the image, the network ensures that the downstream segmentation process can focus on a well-defined region of interest, thereby simplifying subsequent steps.

After obtaining a well-defined bounding region, the second part of the pipeline performs a segmentation task to isolate individual electrode tabs. We employ a watershed image segmentor to first define the bulk battery region and then to segment the tabs. Using a dynamic approach, with various user-controlled sliders, the approach can handle images with variable sizes, varying numbers of tabs, and variable contrasts. Each output channel of the segmentation model corresponds to a single tab, allowing us to extract a 3D array (Num_Tabs × Height × Width) of binary masks, where each mask highlights one tab at a time.

By carefully crafting each stage—first bounding box prediction, then targeted segmentation—we achieve a flexible and robust solution. The result is a system that can handle highly variable imagery: from the subtle differences in electrode tab shapes and arrangements to the diverse image resolutions and scales encountered in X-ray data. Our approach enables straightforward post-processing, visualization, and quantitative analysis of each tab in the electrode stack.

Looking ahead, this framework can be further extended and refined. While the current pipeline focuses on bounding box regression and binary segmentation, future iterations could incorporate more advanced instance segmentation techniques, improved attention modules, or additional shape priors. For now, this two-part solution provides a strong starting point for automated electrode tab detection and segmentation in complex X-ray imagery.

**Note:** This work was developed during the 2-days [Microscopy Hackathon](https://kaliningroup.github.io/mic-hackathon/).

[Video Summary](https://solidpower-my.sharepoint.com/:v:/g/personal/forrest_laskowski_solidpowerbattery_com/ERW033EM8uBIvnGpVUxXA7MBnzIrE-NSe4fnD5X-yNcFeA?e=kuxrcg)

---

## 🚀 Overview
CT4Batt is designed to streamline the analysis of complex X-ray imagery with a focus on electrode tab detection and segmentation. The pipeline operates in two distinct stages:

1. **Bounding Box Prediction**: 
   - Employs a specialized CNN with advanced attention mechanisms.
   - Uses coordinate attention, sliding window attention, and residual feature extractors.
   - Outputs bounding boxes for regions of interest.

2. **Segmentation**: 
   - Converts the zoomed-in bounding region into a series of probability maps for individual tabs.
   - Handles images of varying sizes and tab counts dynamically.
   - Produces binary masks for each tab, facilitating downstream analysis.

---

## 📊 Results

For the zoom-in Contextual CNN models: 
<table>
  <tr>
    <th>Model</th>
    <th>Model A</th>
    <th>Model B</th>
    <th>Model C</th>
  </tr>
  <tr>
    <td>Approach</td>
    <td>CNN with KAN</td>
    <td>CNN with MLP</td>
    <td>Reduced CNN with KAN</td>
  </tr>
  <tr>
    <td>Number of Parameters</td>
    <td>23326976</td>
    <td>4442884</td>
    <td>3646976</td>
  </tr>
  <tr>
    <td>Loss (%) at 200 Epoch</td>
    <td>1.84%</td>
    <td>1.33%</td>
    <td>8.25%</td>
  </tr>
  <tr>
    <td>Average time/epoch (sec)</td>
    <td>0.42</td>
    <td>0.4</td>
    <td>0.4</td>
  </tr>
</table>

Obtained Predictions from Test Dataset: 
![Result Image](evaluation/Model_A.png) 
![Result Image](evaluation/Model_B.png) 
![Result Image](evaluation/Model_C.png) 

---

## 📂 Dataset
We utilized the **X-ray-PBD dataset**, which includes a variety of X-ray images of electrode stacks. The dataset features:
- High variability in tab shapes, arrangements, and resolutions.
- Images from different scales and equipment settings.
---

## ⚙️ How to use Bounding Box Prediction
Follow these steps to get started:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your_username/CT4Batt.git
   cd CT4Batt
   ```

2. **Prepare the Dataset**
   - Download the X-ray-PBD dataset

3. **Train the Model * Predict**
   - Place the path to the dataset in evaluation/base.py file
   - For training and prediction:
   ```bash
   python base.py
   ```
---

## ⚙️ How to use Watershed Segmentation of Tabs
Follow these steps to get started:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your_username/CT4Batt.git
   cd CT4Batt
   ```

2. **Prepare the Dataset**
   - Download the X-ray-PBD dataset, placing images into segmentor/images_repo and masks into segmentor/masks_repo
   - Run the random_image_selector.inypb helper to randomly select images for analysis

3. **Run ct_segmention.py**
   - On the first screen, use the watershed segmentation toggles to select the bulk battery, without the tabs
   - On the second screen, use right/left-click-and-drag to help isolate the tabs from each other
   - Upon pressing confirm a numpy array of all the tab pixels is saved in the results folder, under the segmentor directory
---

## 📚 References
1. **X-ray-PBD Dataset**: [Dataset link](https://github.com/Xiaoqi-Zhao-DLUT/X-ray-PBD?tab=readme-ov-file)
2. **Source for KAN layer code**: [KAN](https://github.com/jakariaemon/CNN-KAN/blob/main/cnn_KAN.py)

---

## 📷 Visualizations
Below are some examples of bounding box prediction and segmentation results:

### Bounding Box Prediction
![Bounding Box Result](evaluation/Model_B.png)

### Watershed Segmentation
![Bulk Battery Isolation](watershed1.PNG)

![Tab Segmentation](watershed2.PNG)
---

## 🌟 Future Work
This framework can be extended to:
- Incorporate advanced instance segmentation techniques.
- Enhance attention modules for improved feature extraction.
- Add shape priors for better performance on batteries with complex geometries.

---

## 🤝 Contributing
We welcome contributions! Please feel free to submit issues or pull requests to improve the project.

---

## 📝 License
This project is licensed under the [MIT License](LICENSE).

---

## 👩‍💻 Authors
**Amir Taqieddin** and **Forrest Laskowski**

**Affilation**: Solid Power Operating Inc., Louisville, CO 80027
