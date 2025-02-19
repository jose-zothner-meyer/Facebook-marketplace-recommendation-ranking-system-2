# Facebook Marketplace Recommendation & Ranking System

## Table of Contents
- [Facebook Marketplace Recommendation \& Ranking System](#facebook-marketplace-recommendation--ranking-system)
  - [Table of Contents](#table-of-contents)
  - [**📌 Project Overview**](#-project-overview)
  - [File Structure](#file-structure)
  - [**📌 Project Roadmap**](#-project-roadmap)
    - [\*\* 1. Data Collection \& Preprocessing\*\*](#-1-data-collection--preprocessing)
    - [\*\* 2. Feature Extraction\*\*](#-2-feature-extraction)
    - [\*\* 3. Search Index \& Ranking System\*\*](#-3-search-index--ranking-system)
    - [\*\* 4. Real-Time Deployment\*\*](#-4-real-time-deployment)
    - [\*\* 5. API Configuration \& Docker Deployment\*\*](#-5-api-configuration--docker-deployment)
  - [**🚀 Technologies \& Tools**](#-technologies--tools)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Clone the Repository](#clone-the-repository)
    - [Create and Activate a Virtual Environment](#create-and-activate-a-virtual-environment)
    - [Install Dependencies](#install-dependencies)
  - [Usage](#usage)
    - [1. Data Processing, Dataset Inspection, and Model Training](#1-data-processing-dataset-inspection-and-model-training)
    - [2. Extract Image Embeddings](#2-extract-image-embeddings)
    - [3. Process a Single Image](#3-process-a-single-image)
    - [4. Similarity Search Using FAISS](#4-similarity-search-using-faiss)
    - [5. Run the API with Docker](#5-run-the-api-with-docker)
  - [Additional Notes](#additional-notes)

---

## **📌 Project Overview**
This project implements an end-to-end machine learning pipeline to recreate a similarity search system inspired by Facebook Marketplace. It processes raw product and image data, trains a fine-tuned ResNet50 Convolutional Neural Network (CNN) for image feature extraction, builds a FAISS (Facebook AI Similarity Search) index for efficient similarity search, and deploys the system as a scalable API using FastAPI and Docker on AWS EC2. The core goal is to enhance product discovery on platforms like Facebook Marketplace by leveraging both image and text data to improve **product ranking & recommendations**.

---

## File Structure

- **`app/`**
  Contains files related to the API and Docker deployment:
    -  **`Dockerfile`**:  Defines the Docker image for containerizing the FastAPI application.
    -  **`main.py`**:  Likely contains the FastAPI application code, defining API endpoints for model serving.
    -  **`requirements.txt`**:  Lists Python dependencies specifically for the API application (if separate from the main project).

- **`data_cleaner.py`**
  Handles cleaning and preprocessing of the raw product and image data. This script likely contains functions to standardize and format the initial dataset.

- **`data_loader.py`**
  Responsible for loading the dataset, including product information and images, from storage into the project for processing and model training.

- **`dataset_processor.py`**
  Processes the loaded dataset to prepare it for model training. This might include tasks like data splitting, batching, and applying transformations.

- **`faiss_search.py`**
  Implements the FAISS (Facebook AI Similarity Search) functionality. This script builds the FAISS index from image embeddings and performs similarity searches based on query images.

- **`file_handler.py`**
  Manages file input and output operations within the project. This could include saving and loading model weights, embeddings, and other project artifacts.

- **`image_processor.py`**
  Focuses on processing individual images, likely for tasks such as visualizing model input or extracting embeddings for single images.

- **`main.py`**
  The main entry point for the project. This script likely orchestrates the entire pipeline, from data loading and preprocessing to model training and evaluation.

- **`model.py`**
  Contains the definition of the main machine learning model, potentially including the ResNet50 adaptation and any custom layers or modifications.

- **`pipeline.py`**
  Defines and manages the overall machine learning pipeline workflow. This script might organize the execution of different stages, such as data processing, model training, and deployment.

- **`resNet.py`**
  Specifically focuses on the ResNet model architecture. This script likely contains the code for loading, adapting, and utilizing the ResNet50 model for feature extraction.

---

## **📌 Project Roadmap**
This project involves multiple stages, from **data acquisition and preprocessing to model training, similarity search implementation, and real-time deployment**. Below is a detailed implementation plan:

### ** 1. Data Collection & Preprocessing**
✅ **Step 1: Tabular Data Cleaning**
- **Data Retrieval:** Accessed a dataset hosted on AWS EC2, requiring SSH access using a PEM file from an S3 bucket.
- **Dataset Exploration:** Explored a dataset containing over 12,500 images and CSV files (product information and image data).
- **Product Data Standardization:** Standardised **product listings** (e.g., prices, categories, locations) from `Cleaned_Products.csv`.
- **Text Feature Extraction:**  Extracted **text features** from product descriptions within the product CSV.
- **Category Encoding:** Encoded product categories from the product CSV into numerical labels for supervised learning and saved a decoder for label translation.

✅ **Step 2: Image Preprocessing**
- **Image Uniformity:** Cleaned images to ensure uniformity in size and channels.
- **Resizing:** Resized images to **256x256** pixels to standardise input dimensions for the CNN.
- **Color Channel Conversion:** Converted all images to **RGB** format to ensure consistent color representation.
- **Data Augmentation (Planned):**  Planned to apply **data augmentation** techniques (e.g., random rotation, flips) to enhance model generalization during training (implementation included in training pipeline).
- **Dataset Labeling:** Joined product category information onto the image CSV (`Images.csv`) using product IDs, enabling image labeling based on filenames and reducing the labeled image dataset to approximately 8000 images.

✅ **Step 3: Merging Data**
- **Combined Dataset Creation:** Linked images with their corresponding product descriptions and category labels.
- **Single Dataset Structure:** Created a **single dataset** that contains both **images**, **text features** (product descriptions), and **category labels** for model training and evaluation.

---

### ** 2. Feature Extraction**
✅ **Step 4: Image Embeddings (CNN)**
- **Pre-trained Model Selection:** Used a **pre-trained ResNet-50** CNN architecture for image feature extraction, leveraging its learned representations from large image datasets.
- **Model Adaptation:** **Adapted the ResNet-50 model** by modifying the final classification layer to match the number of categories in the dataset, ensuring compatibility with the classification task.
- **Transfer Learning Implementation:** Applied **transfer learning** by **unfreezing the last two layers of the adapted ResNet-50 model** to fine-tune it on the Marketplace product categories, allowing the model to learn task-specific image features.
- **Feature Extractor Conversion:** Converted the trained ResNet-50 model into a **feature extractor** by removing the final classification layer, enabling the extraction of 1000-dimensional image embeddings.

✅ **Step 5: Text Embeddings (CNN)**
- **Text Vectorization (Planned):** Planned to convert text descriptions into vector form.
  - **CNN for Text (Planned):**  Planned to utilize a **CNN** for local feature extraction from text descriptions to capture relevant textual information (implementation not fully realized in current version, focusing on image embeddings).

✅ **Step 6: Multi-Modal Fusion (Image Focus in Current Version)**
- **Multi-Modal Embedding (Image Focus):** While the roadmap initially included multi-modal fusion, the current implementation primarily focuses on **image embeddings** generated by the ResNet-50 model for similarity search.
- **Shallow Feedforward Network (Planned):**  Planned to train a **shallow feedforward network** to classify products based on combined image and text embeddings (not implemented in the current version, future enhancement).

---

### ** 3. Search Index & Ranking System**
✅ **Step 7: Create the Search Index**
- **Vector Database Selection:** Chose **FAISS (Facebook AI Similarity Search)** as the vector database for storing and efficiently searching image embeddings.
- **FAISS Index Building:** Stored **product (image) embeddings** generated by the ResNet-50 feature extractor in a **FAISS index**.
- **Approximate Nearest Neighbors (ANN):** Utilized **FAISS Approximate Nearest Neighbors (ANN)** algorithm for fast and scalable retrieval of similar embeddings.
- **Image ID Indexing:** Used image IDs as indices within the FAISS model to enable retrieval of similar images based on vector searches.

✅ **Step 8: Query Embeddings**
- **Query Embedding Extraction:** Converted **query images** into embeddings using the trained **ResNet-50 feature extractor model**.
- **Text Query Embedding (Planned):** Planned to convert **user search queries** into embeddings using a trained **text model** (not implemented in current version, focusing on image-based similarity search).
- **Embedding Retrieval:** Retrieved top **K** nearest product embeddings from the FAISS search index based on the query image embedding.

✅ **Step 9: Final Ranking (Similarity-Based Ranking in Current Version)**
- **Similarity-Based Ranking:** In the current version, ranking is primarily based on **similarity scores** derived from the FAISS vector search, effectively ranking images by visual similarity.

---

### ** 4. Real-Time Deployment**
✅ **Step 10: Deploy as an API**
- **API Framework Selection:** Selected **FastAPI**, a modern Python web framework, to build API endpoints for the FAISS model, leveraging its asynchronous support and automatic API documentation.
- **Containerization:** Containerized the **ranking system (FAISS model and FastAPI application)** using **Docker** to ensure consistency and portability across deployment environments.
- **Cloud Deployment:** Deployed the **Docker image** containing the FastAPI application and FAISS model to an **AWS EC2 instance**, utilizing scalable cloud infrastructure.
- **API Configuration:** Configured the **AWS EC2 instance** with necessary security settings, networking configurations, and access controls to ensure secure and reliable API operation.
- **Low-Latency Optimization:** Optimized the deployment for **low-latency inference** to provide fast similarity search results via the API.

---

### ** 5. API Configuration & Docker Deployment**
✅ **Step 12: FastAPI API Implementation**
- **API Endpoint Definition:** Developed API endpoints using FastAPI within the `app/main.py` script to expose the FAISS similarity search functionality.
- **Request Handling:** Implemented request handling logic within the API to receive query images and return similar image results.
- **API Documentation:** Leveraged FastAPI's automatic documentation generation to provide API documentation for ease of use and integration.

✅ **Step 13: Docker Containerization**
- **Dockerfile Creation:** Created a `Dockerfile` within the `app/` directory to define the Docker image for the FastAPI application and its dependencies.
- **Dependency Management:** Managed API dependencies using a `requirements.txt` file within the `app/` directory, ensuring a consistent and reproducible environment.
- **Image Building:** Built a Docker image containing the FastAPI application, FAISS model, and all necessary dependencies.

✅ **Step 14: AWS EC2 Deployment**
- **Cloud Instance Provisioning:** Provisioned an AWS EC2 instance to host the Dockerized API application.
- **Docker Deployment to EC2:** Deployed the Docker image to the provisioned AWS EC2 instance, enabling cloud-based API access.
- **API Access Configuration:** Configured networking and security settings on the AWS EC2 instance to allow secure and reliable access to the deployed API.
- **Scalability Considerations:** Designed the deployment architecture with scalability in mind, leveraging Docker and AWS EC2 for potential horizontal scaling in the future.

---

## **🚀 Technologies & Tools**
| **Component**       | **Technology Used** | **Details**                                                                 |
|---------------------|--------------------|-----------------------------------------------------------------------------|
| **📊 Data Processing**  | `pandas`           | For tabular data manipulation and analysis.                                  |
|                     | `numpy`            | For numerical operations and array handling.                                  |
|                     | `scikit-learn`     | For label encoding and data splitting.                                      |
| **🖼️ Image Processing**  | `PIL`              | Python Imaging Library for image manipulation.                               |
|                     | `torchvision`      | PyTorch's library for image transformations and dataset handling.             |
| **🧠 Deep Learning**  | `PyTorch`          | Deep learning framework used for model building and training.                |
|                     | `ResNet-50 CNN`    | Pre-trained Convolutional Neural Network adapted for image feature extraction. |
|                     | `TensorBoard`      | Visualization toolkit for monitoring model training progress.                 |
| **📡 Search Indexing**  | `FAISS`            | Facebook AI Similarity Search for efficient vector similarity search.        |
| **🖥️ Deployment**  | `Docker`           | Containerization platform for packaging and deploying the application.       |
|                     | `FastAPI`          | Modern, fast web framework for building the API.                              |
|                     | `AWS EC2`          | Amazon Web Services Elastic Compute Cloud for cloud deployment.              |
|                     | `Uvicorn`          | Asynchronous web server for running the FastAPI application.                  |

---

This project is a **real-world application** of **multi-modal AI + search indexing + cloud deployment**.
It demonstrates a practical approach to building a scalable similarity search system, drawing inspiration from real-world applications like Facebook Marketplace.
Let me know if you need help with specific components or have further questions! 🚀🔥

---

## Installation

### Prerequisites

- **Python 3.13+**
- A virtual environment manager (e.g., Conda or venv)
- Required Python packages (see `requirements.txt`)
- **Docker** (for API deployment)
- **AWS Account** (for cloud deployment - optional, API can be run locally with Docker)

### Clone the Repository

```bash
git clone [https://github.com/your-username/your-repository.git](https://github.com/your-username/your-repository.git)
cd your-repository
```

### Create and Activate a Virtual Environment

Using Conda:
```bash
conda create --name marketplace_env python=3.13.1
conda activate marketplace_env
```

Or using venv:
```bash
python -m venv marketplace_env
source marketplace_env/bin/activate   # On Unix/macOS
marketplace_env\Scripts\activate      # On Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
pip install -r app/requirements.txt # Install API dependencies (if separate)
```

---

## Usage

### 1. Data Processing, Dataset Inspection, and Model Training

Run the main script to process raw data, inspect the dataset, and train the transfer learning model:

```bash
python main.py
```

**What It Does:**

- Orchestrates the entire data processing and model training pipeline.
- Utilizes `data_loader.py` to load the dataset.
- Employs `data_cleaner.py` and `dataset_processor.py` for data cleaning and preparation.
- Leverages `resNet.py` and `model.py` to define and train the ResNet50-based model.
- Saves model weights and training metrics.
- May include dataset inspection and visualization steps.

### 2. Extract Image Embeddings

After training, extract embeddings for every image using the trained feature extractor model:

```bash
python image_processor.py
```

**What It Does:**

- Loads the trained feature extractor model defined in `model.py` and `resNet.py`.
- Uses `data_loader.py` to load images.
- Processes images to extract embeddings using the model.
- Saves the extracted image embeddings for use in similarity search.

### 3. Process a Single Image

To process a single image and demonstrate the image processing pipeline:

```bash
python image_processor.py <image_path>
```

**What It Does:**

- Uses `data_loader.py` to load a single image from the specified `<image_path>`.
- Applies image preprocessing steps defined in `dataset_processor.py`.
- Passes the processed image through the feature extraction model from `model.py` and `resNet.py`.
- Outputs the processed image tensor or embedding (depending on script configuration).

### 4. Similarity Search Using FAISS

To perform a similarity search with a query image using the FAISS index:

```bash
python faiss_search.py <query_image_path>
```

**What It Does:**

- Utilizes `file_handler.py` to load pre-calculated image embeddings and the FAISS index.
- Uses `data_loader.py` to load the query image from the specified `<query_image_path>`.
- Employs `image_processor.py` to extract the embedding for the query image.
- Performs a similarity search using the FAISS index to find similar images.
- Prints or displays the results of the similarity search, including similar image IDs and distances.

### 5. Run the API with Docker

To deploy and run the API using Docker:

```bash
cd app
docker build -t marketplace-api .
docker run -p 8000:8000 marketplace-api
```

**What It Does:**

- **`docker build -t marketplace-api .`**:  Navigates to the `app/` directory, builds a Docker image named `marketplace-api` using the `Dockerfile` in that directory. This command packages the FastAPI application and its dependencies into a Docker image.
- **`docker run -p 8000:8000 marketplace-api`**: Runs the Docker image `marketplace-api`. The `-p 8000:8000` flag maps port 8000 on your host machine to port 8000 inside the Docker container, allowing you to access the API on `http://localhost:8000` or `http://127.0.0.1:8000`.
- Once the Docker container is running, the FastAPI API will be accessible, and you can send requests to the API endpoints defined in `app/main.py` (e.g., for performing similarity searches via the API).

---

## Additional Notes

- **Model Weights:**
  The project relies on saved model weights for the image feature extraction model. Ensure that you train the model using `main.py` to generate and save these weights in the designated location for subsequent steps to function correctly.

- **Folder Structure:**
  - Place raw data files in the appropriate data directories as expected by `data_loader.py` and `data_cleaner.py`.
  - Ensure image data is accessible to `data_loader.py` and `image_processor.py` based on the project's data loading conventions.
  - Model weights and any other persistent project artifacts will be managed by `file_handler.py` and saved in designated directories.
  - API related files (Dockerfile, API application code) are located in the `app/` directory.

- **Customization:**
  For customization of model architecture, training hyperparameters, or data processing steps, modify the relevant scripts: `model.py`, `resNet.py`, `dataset_processor.py`, and `pipeline.py`.  For API specific configurations, see files within the `app/` directory.

- **Modular Design:**
  The project is structured into modular Python scripts (`data_loader.py`, `data_cleaner.py`, `model.py`, `faiss_search.py`, etc.) to promote code organization, reusability, and maintainability. Each script focuses on a specific aspect of the pipeline. The API deployment is also modularized within the `app/` directory.