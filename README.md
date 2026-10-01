Dermatological AI System (MELIA)

Overview

This repository contains the core application and backend logic for an automated dermatological triage system. The project leverages Deep Learning and Computer Vision to analyze skin lesions, combining image processing with clinical metadata for a multimodal diagnostic approach.

This system was built to demonstrate proficiency in Machine Learning pipeline integration, backend routing, and interactive frontend development.

Privacy note: For security and compliance with data privacy standards, all sensitive databases (db_pacientes, db_medicos) and proprietary patient metadata have been strictly excluded from this public repository.


Key features

Multimodal Machine Learning: Integrates image classification and clinical data processing using advanced architectures (ResNet50, Vision Transformers).

Interactive dashboard (aplicacion.py): User-friendly interface built for medical personnel to upload images and view AI-generated preliminary assessments.

Backend QR integration (servidor_qr.py): Dedicated backend module to handle QR code generation and reading for efficient patient/doctor data retrieval.

Secure data handling: Modular architecture designed to connect with external, secure MySQL databases rather than relying on local, unsecured files.


## Tech stack

Language: Python 3.x

Machine Learning / AI: TensorFlow / Keras (ResNet50, ViT)

Frontend/UI: Streamlit (via aplicacion.py)

Backend/API: Flask / Requests (via servidor_qr.py)

Data processing: Pandas, NumPy


## Installation & setup

To run this project locally, clone the repository and install the dependencies:

# Clone the repository
git clone https://github.com/erickrodriguez2704-del/MELIA.git

# Navigate to the directory
cd MELIA

# Install required dependencies
pip install -r requirements.txt


Usage

Start the main application dashboard:

streamlit run aplicacion.py

To run the standalone QR server module:

python servidor_qr.py


Model architecture note

The system was designed to utilize heavy weights files (e.g., resnet50_multimodal_final.keras and vit_multimodal_v2.h5). While the architecture and loading scripts are present in the code, the actual weight files are managed via Git LFS or kept locally due to size constraints.


License

This project is licensed under the MIT License - see the LICENSE file for details.
