🏢 HR Analytics: Employee Job Change Prediction & Insights
An end-to-end data science project analyzing workforce data to predict employee turnover and uncover key drivers behind job changes.

📋 Table of Contents
About the Project

Dataset & Features

Methodology & Pipeline

Exploratory Data Analysis (EDA)

Model Training & Evaluation

Key Insights & Findings

Tech Stack

Project Structure

Getting Started

License

🎯 About the Project
Understanding why talented employees leave an organization is crucial for strategic human resource planning and talent management. In this project, we built a comprehensive analytical and predictive pipeline to determine the likelihood of an employee looking for a new job based on their demographic background, educational history, professional experience, and current company environment. Our goal is to help organizations transition from reactive turnover management to proactive workforce retention through data-driven decision-making.

📊 Dataset & Features
The dataset used in this analysis contains detailed records of employees, categorized into several descriptive and numerical attributes:

Demographics: City development index, gender, and relevant work experience.

Background & Education: Enrolled university status, highest education level achieved, and major academic discipline.

Professional Experience: Total years of experience, history of changing previous companies, and employment gaps.

Company Profile: Current company type (e.g., startup, private, public, MNC) and company size (number of employees).

Target Variable: target (1 = Looking for a job change, 0 = Not looking for a job change).

🔄 Methodology & Pipeline
We structured our workflow to ensure data integrity, robust statistical validation, and high predictive accuracy:

Data Preprocessing & Cleaning: We systematically handled missing values using advanced imputation techniques (such as KNN imputation), treated outliers, and standardized data types across all features.

Exploratory Data Analysis (EDA): We visualized complex distributions, evaluated feature correlations using heatmaps, and isolated high-risk attrition segments using Matplotlib and Seaborn.

Feature Engineering & Transformation: We scaled numerical variables to standard distributions and applied one-hot/label encoding to transform categorical attributes into machine-learning-ready formats.

Model Building & Validation: We trained multiple classification algorithms, tuned hyperparameters to optimize performance, and validated results using robust cross-validation strategies.

📈 Exploratory Data Analysis (EDA) Highlights
During our exploratory phase, we deep-dived into several core relationships within the workforce data:

Socio-Economic Impact: Analyzed how the economic status of a city (City Development Index) inversely correlates with employee retention rates.

Experience Tenure: Mapped how years of experience interact with job-hopping tendencies across different career stages.

Educational Triggers: Examined whether specific educational disciplines or university enrollments create higher aspirations for career transitions.

⚙️ Model Training & Evaluation
We evaluated several classification models to predict employee turnover accurately:

Models Tested: Logistic Regression, K-Nearest Neighbors (KNN), Decision Trees, and Random Forest classifiers.

Evaluation Metrics: We assessed our models using Precision, Recall, F1-Score, and ROC-AUC curves to minimize false negatives in identifying at-risk employees.

Handling Imbalance: We applied techniques like SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance in the target variable, ensuring reliable predictive performance.

💡 Key Insights & Findings
City Development Index: Employees residing in lower development index cities consistently showed a significantly higher rate of seeking alternative job opportunities.

Company Type Dynamics: Certain company structures, particularly startups and non-funded organizations, experience much higher mobility and turnover rates compared to stable public institutions or multinational corporations (MNCs).

Experience Thresholds: Mid-level professionals with specific ranges of total experience exhibited distinct peaks in job transition behaviors.

🛠️ Tech Stack
Language: Python

Data Manipulation & Cleaning: Pandas, NumPy

Data Visualization: Matplotlib, Seaborn

Machine Learning & Preprocessing: Scikit-learn, KNN Imputer

📁 Project Structure
Plaintext
📦 HR-Analytics-Job-Change
 ┣ 📂 data                    # Raw and cleaned dataset CSV files
 ┣ 📂 notebooks               # Jupyter notebooks for EDA, preprocessing, and modeling
 ┣ 📂 outputs                 # Generated analytical charts, figures, and model exports
 ┣ 📜 requirements.txt        # Python package dependencies
 ┗ 📜 README.md               # Project documentation
🚀 Getting Started
To set up and run this project locally, follow these steps:

Clone the repository:

Bash
git clone https://github.com/your-username/hr-analytics-job-change.git
Navigate to the project directory:

Bash
cd "Group idea HR Analytics job change"
Install the required dependencies:

Bash
pip install -r requirements.txt
Open the Jupyter notebooks to explore our data analysis and machine learning pipeline step-by-step.
