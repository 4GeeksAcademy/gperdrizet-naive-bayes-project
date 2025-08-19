# Naive Bayes Classifier: Google Play Store Reviews

[![Codespaces Prebuilds](https://github.com/4GeeksAcademy/gperdrizet-naive-bayes-project/actions/workflows/codespaces/create_codespaces_prebuilds/badge.svg)](https://github.com/4GeeksAcademy/gperdrizet-naive-bayes-project/actions/workflows/codespaces/create_codespaces_prebuilds)

A comprehensive machine learning project focused on sentiment analysis of Google Play Store reviews using Naive Bayes classification algorithms. This project demonstrates essential text preprocessing, feature engineering, and model comparison techniques through practical exercises with real-world review data.

## Project Overview

This project analyzes Google Play Store app reviews to classify sentiment polarity (positive/negative) using various Naive Bayes algorithms and advanced dimensionality reduction techniques. The dataset provides hands-on experience with:

- Advanced text preprocessing including lemmatization and filtering
- Natural Language Processing (NLP) techniques with NLTK
- Multiple Naive Bayes classifier variants comparison (including ComplementNB)
- Dimensionality reduction using PCA and Feature Agglomeration
- Hyperparameter optimization with GridSearchCV
- Cross-validation and model evaluation with stratified sampling
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
│   ├── functions.py      # Helper functions for notebooks
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

1. **Advanced Text Preprocessing**: Load, clean, and prepare text data with sophisticated NLP techniques
2. **Text Vectorization**: Convert text reviews into numerical features using CountVectorizer with feature selection
3. **Train-Test Split**: Properly partition data for model training and evaluation
4. **Exploratory Data Analysis**: Analyze review length distributions, filter outliers, and examine text patterns
5. **Model Comparison**: Compare multiple classification algorithms:
   - **Logistic Regression**: Linear baseline model for comparison
   - **Multinomial Naive Bayes**: Ideal for discrete count data (word frequencies)
   - **Gaussian Naive Bayes**: Assumes continuous features with normal distribution
   - **Bernoulli Naive Bayes**: Works with binary/boolean features
   - **Complement Naive Bayes**: Enhanced version of Multinomial NB for imbalanced datasets
6. **Dimensionality Reduction**: Apply PCA and Feature Agglomeration for improved performance
7. **Hyperparameter Optimization**: Use GridSearchCV for automated parameter tuning
8. **Cross-Validation**: Use stratified k-fold cross-validation for robust model evaluation
9. **Performance Visualization**: Create confusion matrices and hyperparameter optimization plots

## Key Features & Techniques

### Text Preprocessing
- **Advanced Text Cleaning**: Lowercase conversion, number and punctuation removal
- **Lemmatization**: Using NLTK's WordNetLemmatizer for word normalization
- **Stop Words Removal**: Filter out common English words using scikit-learn
- **Review Length Filtering**: Remove extremely short and long reviews for data quality
- **Vectorization**: Transform text into numerical word count matrices
- **Feature Selection**: Filter features based on frequency thresholds to reduce noise

### Model Evaluation
- **Stratified Cross-Validation**: 10-fold stratified cross-validation for reliable performance estimates
- **Hyperparameter Optimization**: GridSearchCV for automated parameter tuning
- **Confusion Matrix**: Detailed breakdown of classification performance for multiple models
- **Accuracy Metrics**: Comprehensive evaluation of model effectiveness
- **Pipeline Optimization**: Advanced model pipelines combining dimensionality reduction with classification

### Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Linear dimensionality reduction with variance analysis
- **Feature Agglomeration**: Hierarchical clustering of features to reduce dimensionality
- **Cross-Correlation Analysis**: Spearman correlation analysis of features
- **Pipeline Integration**: Seamless integration of reduction techniques with classifiers

### Exploratory Data Analysis
- **Review Length Analysis**: Distribution analysis with outlier detection and filtering
- **Feature Distribution**: Analysis of word frequency patterns and feature selection
- **Data Quality Assessment**: Systematic identification and handling of data quality issues
- **Visualization**: Comprehensive plotting of distributions and model performance

## Technologies Used

- **Python 3.11**: Core programming language
- **Pandas 2.3.1**: Data manipulation and analysis
- **NumPy 2.3.2**: Numerical computing and array operations
- **Scikit-learn 1.7.1**: Machine learning algorithms, pipelines, and evaluation metrics
- **NLTK**: Natural Language Toolkit for advanced text processing and lemmatization
- **Matplotlib 3.10.3**: Data visualization and plotting
- **Seaborn 0.13.2**: Statistical data visualization
- **SciPy 1.16.1**: Scientific computing and statistical functions
- **Jupyter 1.1.1**: Interactive development environment

## Model Performance

The project compares multiple classification algorithms and advanced techniques:

### Core Algorithms
1. **Logistic Regression**: Linear baseline model for performance comparison
2. **Multinomial Naive Bayes**: Best suited for word count features in text classification
3. **Gaussian Naive Bayes**: Assumes continuous, normally distributed features
4. **Bernoulli Naive Bayes**: Optimized for binary feature representations
5. **Complement Naive Bayes**: Enhanced Multinomial variant designed for imbalanced datasets

### Advanced Techniques
1. **PCA + Complement Naive Bayes**: Dimensionality reduction pipeline with hyperparameter optimization
2. **Feature Agglomeration + Complement NB**: Hierarchical feature clustering with multiple linkage methods
3. **Feature Agglomeration + Multinomial NB**: Alternative clustering approach for comparison

The solution demonstrates systematic hyperparameter optimization using GridSearchCV with stratified cross-validation, providing robust performance estimates and identifying optimal model configurations.

## Contributing

This is an educational project. Contributions for improving the analysis, adding new features, or enhancing documentation are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

