___
# AI System for Lung nodule follow up using plain chest X-Rays
___

This repository contains the code and resources for a research project focused on developing an AI-powered system for detecting and tracking lung nodules using Chest X-Rays (CXRs), supported by synthetic X-ray data derived from CT scans.

### Project Overview
Lung nodules can indicate early-stage lung cancer, but traditional detection using Chest X-Rays is limited in accuracy. This project introduces an AI system trained on Digitally Reconstructed Radiographs (DRRs) generated from CT scans with annotated nodules to enhance detection accuracy and enable reliable follow-up.

### Objectives
- Develop a deep learning model to detect nodules in Chest X-Rays.
- Generate synthetic, labeled training data by projecting nodule positions from CT scans to DRRs.
- Enable progression tracking of nodules via image registration.
- Provide structured, radiologist-reviewable reports

### Proposed Methodology

![image](https://github.com/user-attachments/assets/37ef825a-db49-4ffc-a703-1908d9786455)

1. Data Collection & Anonymization : Anonymize CT scans with identified nodules and corresponding real CXRs.
2. DRR Generation : Convert CT volumes to DRR images mimicking real X-rays using ray casting techniques.
3. Nodule Projection : Map annotated nodules from CT to the DRRs for supervised learning.
4. AI Model Training : Train a deep learning model (e.g., CNN) to detect and localize nodules on annotated DRRs.
5. Follow-up Comparison : Align CXRs over time to detect changes in nodule size/position.
6. Structured Reporting : Generate millimeter-scale, image-annotated reports to assist radiologists.

### Ethical Considerations
All patient data is anonymized. This project strictly follows ethical guidelines, ensuring compliance with hospital data-sharing regulations.

### License
For academic and research use only. Licensing details will be added upon publication approval.
