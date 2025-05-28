---
layout: home
permalink: index.html

# Please update this with your repository name and title
repository-name: e19-4yp-AI-System-for-Lung-nodule-follow-up-using-plain-chest-X-Ray
title: AI System for Lung Nodule Follow-Up Using Plain Chest X-Rays
---

[comment]: # "This is the standard layout for the project, but you can clean this and use your own template"

# Project Title

#### Team

- E/19/366, W.A.M.P. Senevirathne, [email](mailto:e19366@eng.pdn.ac.lk)
- E/19/224, M.M.S.H. Madhurasinghe, [email](mailto:e19224@eng.pdn.ac.lk)
- E/18/059, D.M. De Silva, [email](mailto:e18059@eng.pdn.ac.lk)

#### Supervisors

- Dr. Chathura Weerasinghe
- Mr. B.A.K. Dissanayake

#### Table of content

1. [Abstract](#abstract)
2. [Related works](#related-works)
3. [Methodology](#methodology)
4. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
5. [Predicted Outcomes](#predicted-outcomes)
6. [Conclusion](#conclusion)
7. [Publications](#publications)
8. [Links](#links)

---

## Abstract
Lung nodules, potential early signs of lung cancer, are typically detected via CT scans, but their high cost and radiation exposure limit frequent use. While chest X-rays (CXRs) are more accessible, they often miss small nodules, leading to delayed diagnoses.  

This project aims to develop an AI-powered system for detecting and tracking lung nodules using CXRs, reducing reliance on CT scans. It employs deep learning-based Computer-Aided Detection (CAD) techniques, utilizing Digitally Reconstructed Radiographs (DRRs) to train the AI model for enhanced detection and follow-up. By integrating feature extraction, image registration, and deep learning, the system offers a reliable, automated solution for lung nodule monitoring.  

Validated against real-world datasets and radiologist interpretations, the AI system aims to improve early detection and tracking, aiding clinical decisions while minimizing unnecessary CT referrals. This research advances AI-driven diagnostics, making lung cancer screening more accessible and efficient, especially in resource-limited settings.


## Related works
Early studies highlighted the limitations of chest X-rays (CXRs) in detecting small nodules due to anatomical complexity, leading to a reliance on CT scans for higher sensitivity. However, CT scans are costly and expose patients to radiation, prompting interest in AI-assisted Computer-Aided Detection (CAD) systems to improve CXR-based diagnosis.   

Deep learning techniques, particularly convolutional neural networks (CNNs), have significantly advanced automated nodule detection. Initial AI models focused on preprocessing strategies like lung field segmentation and feature extraction, while later approaches incorporated multi-resolution patch-based CNNs and object detection frameworks such as YOLOv4 to enhance detection accuracy. Studies demonstrated that AI models could complement radiologists, improving sensitivity while maintaining acceptable false-positive rates.   

Efforts also explored AI-driven nodule tracking using follow-up CXRs, with models trained on Digitally Reconstructed Radiographs (DRRs) to simulate CT-derived nodules. AI has been integrated into hospital Picture Archiving and Communication Systems (PACS), facilitating real-time nodule detection and aiding radiologists in monitoring growth patterns based on established clinical guidelines.  

These foundational studies paved the way for AI-powered lung nodule detection, refining methods to make screening more accessible, cost-effective, and efficient in resource-limited settings. Let me know if you’d like further refinements!


## Methodology
This study proposes a hybrid AI-driven approach to enhance lung nodule detection and
tracking by integrating CT scans, Digitally Reconstructed Radiographs (DRRs), and
deep learning models trained on X-ray datasets.


![image](https://github.com/user-attachments/assets/37ef825a-db49-4ffc-a703-1908d9786455)   
1. Data Collection & Anonymization : Anonymize CT scans with identified nodules and corresponding real CXRs.
2. DRR Generation : Convert CT volumes to DRR images mimicking real X-rays using ray casting techniques.
3. Nodule Projection : Map annotated nodules from CT to the DRRs for supervised learning.
4. AI Model Training : Train a deep learning model (e.g., CNN) to detect and localize nodules on annotated DRRs.
5. Follow-up Comparison : Align CXRs over time to detect changes in nodule size/position.
6. Structured Reporting : Generate millimeter-scale, image-annotated reports to assist radiologists.


## Experiment Setup and Implementation

The experimental framework for this project is designed to evaluate the effectiveness of an AI-based CAD system in detecting and tracking lung nodules using plain chest X-rays. The setup consists of several key stages:

* **Hardware Environment**: Experiments will be conducted on GPU-enabled cloud environments (e.g., Google Colab Pro/Pro+, AWS EC2 with NVIDIA T4/A100 GPUs) and local machines equipped with at least 16GB RAM and NVIDIA GTX 1660 or higher.

* **Software Stack**: The implementation stack includes:

  * **PyTorch** and **TensorFlow** for model development.
  * **SimpleITK**, **pydicom**, and **MONAI** for medical image processing.
  * **OpenCV** for general image preprocessing and transformations.
  * **Docker** and **FastAPI** for deployment and API integration.
  * **Matplotlib/Seaborn** for data visualization and report generation.

* **Dataset Sources**:

  * **CT Datasets**: LIDC-IDRI, NLST for DRR generation.
  * **CXR Datasets**: CheXpert, JSRT, and VinDr-CXR for real-world X-ray validation.
  * Images are preprocessed to 512x512 pixels and anonymized using `deid`.



## Predicted Outcomes
- Reduced false negatives in the system we propose
- Optimized nodule feature extraction to assist radiologists

**Potential Benefits**:
- More reliable AI system for Nodule detection
- Help Radiologist
- Low cost for a follow-up
- Less radiation exposure for patients 


## Conclusion
Our research develops an AI-powered system for lung nodule detection and follow-up using chest X-rays, enhancing accessibility and reducing reliance on CT scans. By leveraging deep learning and DRR-based training, our system improves early detection, automates nodule tracking, and supports radiologists in clinical decision-making.

## Publications
[//]: # "Note: Uncomment each once you uploaded the files to the repository"
_not yet published_
<!-- 1. [Semester 7 report](./) -->
<!-- 2. [Semester 7 slides](./) -->
<!-- 3. [Semester 8 report](./) -->
<!-- 4. [Semester 8 slides](./) -->
<!-- 5. Author 1, Author 2 and Author 3 "Research paper title" (2021). [PDF](./). -->


## Links

[//]: # ( NOTE: EDIT THIS LINKS WITH YOUR REPO DETAILS )

- [Project Repository](https://github.com/cepdnaclk/repository-name)
- [Project Page](https://cepdnaclk.github.io/repository-name)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)

[//]: # "Please refer this to learn more about Markdown syntax"
[//]: # "https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet"
