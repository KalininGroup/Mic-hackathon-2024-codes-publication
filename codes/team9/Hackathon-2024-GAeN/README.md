# 🧬 **Automated Nanoparticle Detection in Microscope Images**

## **Overview**  
This project automates the detection and localization of nanoparticles in microscope images, a traditionally manual and time-consuming task. Using **machine learning** models and an intuitive **user interface**, researchers can efficiently analyze samples, saving significant time and effort.

The system includes:  
1. **User-Friendly Interface**: Displays a simulated microscope's current view, detects nanoparticles, and provides real-time results.  
2. **Machine Learning Models**: Predicts the probability of nanoparticles, counts them, and identifies their centers.  
3. **Automated Navigation**: Systematically scans the sample, optimizing microscope movement to locate nanoparticles efficiently.

---

## **Features**  
- **Real-Time Detection**: Identifies nanoparticles, their quantity, and their coordinates from 256x256 pixels microscope image sections.  
- **Scanning**: Automatically navigates through the sample, skipping areas without nanoparticles for maximum efficiency.  
- **Visualization**: Tracks microscope movements and displays results in an intuitive interface.  
- **Synthetic Data Generation**: Creates labeled datasets to train and validate machine learning models.

## **Deploy GUI**

  <p align ="center">
  <img alt="demo" src="Data/demo.gif"/>
  </p>

---

## **Getting Started**  

### Prerequisites  
Before running the code, ensure you have the following installed:  
- Python 3.8 or higher  
- Required Python libraries:  
  ```bash
  pip install numpy pytorch hyperspy atomai matplotlib opencv-python
  ```  
- A working microscope setup (if applicable) or access to images for testing.  


## **Directory Structure**  
```
nanoparticle-detector/
│
├── Data/                      # Input images and labels
│   ├── ML_training_data/      # Original microscope images with nanoparticles        
│   └── simulated_grid.png     # Simulated image
│
├── ML/models                  # Saved machine learning models
│   └── nanos_detector.pkl     # Model to detect nanoparticles (the other model is in a One Drive link to download)
│
├── Auxiliar files/  
│   └── Sample_generetor.py     # Script to generate data
│
├──GUI/
│   ├── navigation.py          # Navigation system
|   ├── auxiliar_functions.py  
|   ├── settings.py
|   └── main.py                # Main user interface with navigator
└── README.md         # This file
```

---

## **How It Works**  
1. **Data Preparation**:  
   - Real and synthetic microscope images are collected and labeled.  
   - Images are split into 128x128 regions for model training.  

2. **Model Training**:  
   - Two machine learning models are trained to:  
     - Predict nanoparticle probability.  
     - Return the coordinetas of the nanoparticle center .

3. **Automated Navigation**:  
   - The microscope window scans a 256x256 image.  
   - When nanoparticles are detected, the navigator focuses on high-probability regions.

4. **User Interface**:  
   - Displays the microscope's current view.  
   - Shows nanoparticle counts, coordinates, and navigation paths in real time.

---

## **Results**  
The system:  
- **Accelerates analysis** by automating nanoparticle detection.  
- **Optimizes microscope usage** with a focused navigation approach.  
- **Provides accurate predictions** with well-trained machine learning models.  

---

## **Acknowledgements**  
This project was developed with the support of advanced electron microscopy groups and researchers. Special thanks to our collaborators and advisors for their invaluable input.  

---

## **Contact**  
Ivan Pinto, Josep Cruañes, Jovan Pomar, Xuli Chen, Marta Torrens
- Advanced Electron Nanoscopy Group, ICN2 
For questions or collaborations, please contact:  
📧 **ivan.pinto@icn2.cat**  
