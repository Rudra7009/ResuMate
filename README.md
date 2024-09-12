
# Resume Screening Project

A Python-based application that utilizes machine learning and natural language processing techniques to automate the resume screening process. This project reads a resume as input and predicts the most suitable job title based on its content. The application uses data from Kaggle for training, and multiple Jupyter notebooks (`.ipynb`) are included for experimentation and analysis.

## Features

- **Automated Resume Parsing**: Extracts key information from resumes in PDF or text format.
- **Job Title Prediction**: Provides a suitable job title based on the content of the resume.
- **Kaggle Dataset**: Utilizes a publicly available dataset from Kaggle for training and validation.
- **Jupyter Notebooks**: Includes various notebooks for data preprocessing, model training, and evaluation.

## Installation

1. Clone the repository:
   git clone https://github.com/your-username/resume-screening-project.git
   cd resume-screening-project
2. Install dependencies:
   pip install -r requirements.txt
3. Download the Kaggle dataset:
   Visit Kaggle and download the dataset used for training.
   Place the dataset in the data/ directory.

## Usage
   
1. Run the Jupyter notebooks to preprocess data and train the models:


jupyter notebook
Open the preprocessing.ipynb and model_training.ipynb notebooks to execute the required steps.

2. To predict the job title for a new resume, use the following Python command:


python predict_job_title.py --resume <path-to-resume>
Example:

python predict_job_title.py --resume sample_resume.pdf
             
