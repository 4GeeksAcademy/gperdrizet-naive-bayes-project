# Naive Bayes Classifier: Google Play Store Reviews

[![Codespaces Prebuilds](https://github.com/4GeeksAcademy/gperdrizet-naive-bayes-project/actions/workflows/codespaces/create_codespaces_prebuilds/badge.svg)](https://github.com/4GeeksAcademy/gperdrizet-naive-bayes-project/actions/workflows/codespaces/create_codespaces_prebuilds)

A comprehensive machine learning project focused on sentiment analysis of Google Play Store reviews using Naive Bayes classification algorithms. This project demonstrates essential text preprocessing, feature engineering, and model comparison techniques through practical exercises with real-world review data.

## Project Overview

This project analyzes Google Play Store app reviews to classify sentiment polarity (positive/negative) using various Naive Bayes algorithms. The dataset provides hands-on experience with:

- Text data preprocessing and vectorization
- Natural Language Processing (NLP) techniques
- Multiple Naive Bayes classifier variants comparison
- Cross-validation and model evaluation
- Confusion matrix analysis and performance metrics
- Exploratory Data Analysis (EDA) for text data

## Getting Started

### Option 1: GitHub Codespaces (Recommended)

1. **Fork the Repository**
   - Click the "Fork" button on the top right of the GitHub repository page
   - 4Geeks students: set 4GeeksAcademy as the owner - 4Geeks pays for your codespace usage. All others, set yourself as the owner
   - Give the fork a descriptive name. 4Geeks students: I recommend including your GitHub username to help in finding the fork if you lose the link
   - Click "Create fork"
   - 4Geeks students: bookmark or otherwise save the link to your fork

2. **Create a GitHub Codespace**
   - On your forked repository, click the "Code" button
   - Select "Create codespace on main"
   - If the "Create codespace on main" option is grayed out - go to your codespaces list from the three-bar menu at the upper left and delete an old codespace
   - Wait for the environment to load (dependencies are pre-installed)

3. **Start Working**
   - Open `notebooks/mvp.ipynb` in the Jupyter interface
   - Follow the step-by-step instructions in the notebook

### Option 2: Local Development

1. **Prerequisites**
   - Git
   - Python >= 3.10

2. **Fork the repository**
   - Click the "Fork" button on the top right of the GitHub repository page
   - Optional: give the fork a new name and/or description
   - Click "Create fork"

3. **Clone the repository**
   - From your fork of the repository, click the green "Code" button at the upper right
   - From the "Local" tab, select HTTPS and copy the link
   - Run the following commands on your machine, replacing `<LINK>` and `<REPO_NAME>`

   ```bash
   git clone <LINK>
   cd <REPO_NAME>
   ```

4. **Set Up Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Launch Jupyter & start the notebook**
   ```bash
   jupyter notebook notebooks/mvp.ipynb
   ```

## Project Structure

```
├── .devcontainer/        # Development container configuration
├── data/                 # Data file directory
├── models/               # Trained model storage directory
│
├── notebooks/            # Jupyter notebook directory
│   ├── mvp.ipynb         # Assignment notebook
│   └── solution.ipynb    # Solution notebook
│
├── .gitignore           # Files/directories not tracked by git
├── LICENSE              # GNU General Public License v3.0
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Dataset

The dataset contains Google Play Store app reviews loaded from a remote URL with the following key features:
- **Review Text**: Raw user review content for sentiment analysis
- **Polarity**: Binary sentiment labels (0 = negative, 1 = positive)

The data is sourced from: `https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv`

**Note**: This dataset was collected for academic purposes only. No commercial benefit was obtained from data collection activities.

## Learning Objectives

1. **Data Preprocessing**: Load and prepare text data for machine learning
2. **Text Vectorization**: Convert text reviews into numerical features using CountVectorizer
3. **Train-Test Split**: Properly partition data for model training and evaluation
4. **Exploratory Data Analysis**: Analyze review length distributions and text patterns
5. **Model Comparison**: Compare three Naive Bayes variants:
   - **Multinomial Naive Bayes**: Ideal for discrete count data (word frequencies)
   - **Gaussian Naive Bayes**: Assumes continuous features with normal distribution
   - **Bernoulli Naive Bayes**: Works with binary/boolean features
6. **Cross-Validation**: Use k-fold cross-validation for robust model evaluation
7. **Performance Visualization**: Create confusion matrices and performance plots
8. **Baseline Comparison**: Compare against simple baseline models

## Key Features & Techniques

### Text Preprocessing
- **Stop Words Removal**: Filter out common English words using scikit-learn
- **Vectorization**: Transform text into numerical word count matrices
- **Feature Engineering**: Extract meaningful features from raw text data

### Model Evaluation
- **Cross-Validation**: 7-fold cross-validation for reliable performance estimates
- **Confusion Matrix**: Detailed breakdown of classification performance
- **Accuracy Metrics**: Comprehensive evaluation of model effectiveness
- **Baseline Comparison**: Performance comparison with simple heuristic models

### Exploratory Data Analysis
- **Review Length Analysis**: Distribution of review character counts
- **Feature Distribution**: Analysis of word frequency patterns
- **Data Quality Assessment**: Identification of outliers and data quality issues

## Technologies Used

- **Python 3.11**: Core programming language
- **Pandas 2.3.1**: Data manipulation and analysis
- **NumPy 2.3.2**: Numerical computing and array operations
- **Scikit-learn 1.7.1**: Machine learning algorithms and evaluation metrics
- **Matplotlib 3.10.3**: Data visualization and plotting
- **Seaborn 0.13.2**: Statistical data visualization
- **Jupyter 1.1.1**: Interactive development environment

## Model Performance

The project compares three Naive Bayes variants:

1. **Multinomial Naive Bayes**: Best suited for word count features in text classification
2. **Gaussian Naive Bayes**: Assumes continuous, normally distributed features
3. **Bernoulli Naive Bayes**: Optimized for binary feature representations

Cross-validation results show the relative performance of each algorithm, with detailed accuracy metrics and statistical significance testing.

## Contributing

This is an educational project. Contributions for improving the analysis, adding new features, or enhancing documentation are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

