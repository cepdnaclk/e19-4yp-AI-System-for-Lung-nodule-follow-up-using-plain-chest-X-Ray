___
# AI System for Lung nodule follow up using plain chest X-Rays
___

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Academic-lightgrey)](#license)
[![Issues](https://img.shields.io/github/issues/cepdnaclk/e19-4yp-AI-System-for-Lung-nodule-follow-up-using-plain-chest-X-Ray)](https://github.com/cepdnaclk/e19-4yp-AI-System-for-Lung-nodule-follow-up-using-plain-chest-X-Ray/issues)
[![Last Commit](https://img.shields.io/github/last-commit/cepdnaclk/e19-4yp-AI-System-for-Lung-nodule-follow-up-using-plain-chest-X-Ray)](https://github.com/cepdnaclk/e19-4yp-AI-System-for-Lung-nodule-follow-up-using-plain-chest-X-Ray/commits/main)


This repository contains the code and resources for a research project focused on developing an AI-powered system for detecting and tracking lung nodules using Chest X-Rays (CXRs), supported by synthetic X-ray data derived from CT scans.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Proposed Methodology](#proposed-methodology)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Ethical Considerations](#ethical-considerations)
- [Contributing](#contributing)
- [Issues](#issues)
- [License](#license)
- [Contact](#contact)

---

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

## Ethical Considerations
All patient data is anonymized. This project strictly follows ethical guidelines, ensuring compliance with hospital data-sharing regulations.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes. See [CONTRIBUTING.md](docs/CONTRIBUTING.md) (if available) for guidelines.

## Issues

If you encounter any problems, please open an [issue](https://github.com/cepdnaclk/e19-4yp-AI-System-for-Lung-nodule-follow-up-using-plain-chest-X-Ray/issues).

## License
For academic and research use only. Licensing details will be added upon publication approval.

## Contact

For questions or collaborations, please contact the project maintainers via [GitHub Issues](https://github.com/cepdnaclk/e19-4yp-AI-System-for-Lung-nodule-follow-up-using-plain-chest-X-Ray/issues).
