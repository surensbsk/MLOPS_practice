# Machine Learning Project: Social Network Ads

This project is designed to build a machine learning model that predicts user behavior based on data from social network ads. The project is structured into several packages for data processing, model training, a user interface, and an API for deployment.

## Project Structure

```
ml-project
├── data
│   └── raw
│       └── Social_Network_Ads.csv
├── src
│   ├── data_processing
│   │   ├── data_validation.py
│   │   └── preprocessing.py
│   ├── training
│   │   └── train.py
│   └── __init__.py
├── ui
│   ├── app.py
│   └── requirements.txt
├── api
│   ├── main.py
│   └── requirements.txt
├── docker
│   ├── Dockerfile.api
│   ├── Dockerfile.ui
│   └── docker-compose.yml
├── deployment
│   ├── api-deployment.yaml
│   └── ui-deployment.yaml
├── .dvc
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup Instructions

1. **Clone the Repository**
   Clone this repository to your local machine using:
   ```
   git clone <repository-url>
   ```

2. **Install Dependencies**
   Navigate to the project directory and install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. **Data Versioning with DVC**
   Initialize DVC to manage your data:
   ```
   dvc init
   dvc add data/raw/Social_Network_Ads.csv
   ```

4. **Docker Setup**
   Build the Docker images for the API and UI:
   ```
   docker-compose build
   ```

5. **Run the Application**
   Start the services using Docker Compose:
   ```
   docker-compose up
   ```

6. **Access the UI**
   Open your web browser and go to `http://localhost:8501` to access the Streamlit UI.

7. **API Access**
   The API can be accessed at `http://localhost:8000` for making predictions.

## Usage

- **Data Processing**: The `src/data_processing` package contains scripts for validating and preprocessing the data.
- **Model Training**: The `src/training/train.py` script is responsible for training the machine learning model.
- **User Interface**: The Streamlit app in `ui/app.py` allows users to input data and view predictions.
- **API**: The API in `api/main.py` serves the model and allows for HTTP requests to make predictions.

## Deployment

Deployment configurations for Kubernetes are provided in the `deployment` directory. Use the following commands to deploy the API and UI:

```
kubectl apply -f deployment/api-deployment.yaml
kubectl apply -f deployment/ui-deployment.yaml
```

## Version Control

This project uses Git for version control. Make sure to commit your changes regularly and push them to the remote repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.