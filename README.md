# Student-Performance-Prediction
Machine Learning project to predict student academic performance using Python and Scikit-learn.
# Student Performance Prediction (Machine Learning Project)

## Overview

The Student Performance Prediction project focuses on predicting students' academic performance using Machine Learning techniques. Educational institutions generate large amounts of data related to student activities, study habits, and academic records. Analyzing this data can help identify patterns that influence student performance.

This project uses a dataset containing information about students such as age, study time, parental education, extracurricular activities, and other factors. Using this information, a machine learning model is trained to predict the student's grade class.

The objective of this project is to demonstrate the use of data preprocessing, machine learning algorithms, and evaluation metrics to build an effective predictive model.

---

## Objectives

The main objectives of this project are:

* To analyze student-related data and identify factors affecting academic performance.
* To preprocess and clean the dataset for machine learning analysis.
* To build a machine learning model capable of predicting student grade classes.
* To evaluate the model performance using accuracy metrics.
* To demonstrate the practical application of machine learning in the education domain.

---

## Dataset Information

The dataset contains various attributes related to student academic and personal information.

### Dataset Features

* StudentID – Unique identifier for each student
* Age – Age of the student
* Gender – Student gender
* Ethnicity – Student ethnicity category
* ParentalEducation – Education level of parents
* StudyTimeWeekly – Weekly study hours
* Absences – Number of school absences
* Tutoring – Whether the student receives tutoring
* ParentalSupport – Level of parental support
* Extracurricular – Participation in extracurricular activities
* Sports – Participation in sports
* Music – Participation in music activities
* Volunteering – Participation in volunteering activities
* GPA – Grade Point Average
* GradeClass – Target variable representing student performance class

---

## Technologies Used

This project is implemented using the following technologies:

* Python
* Pandas
* NumPy
* Scikit-learn
* Kaggle Notebook Environment

The dataset was processed and analyzed using Python libraries, while the machine learning model was implemented using Scikit-learn.

---

## Machine Learning Workflow

The following steps were followed to build the model:

### 1. Data Collection

The dataset was obtained from Kaggle and contains structured information about student performance indicators.

### 2. Data Preprocessing

Data preprocessing involved:

* Handling categorical variables
* Converting text features into numerical values using Label Encoding
* Removing unnecessary columns
* Separating input features and target variables

### 3. Feature Selection

Input features were separated from the target variable:

* X – Input features
* y – Target variable (GradeClass)

### 4. Train-Test Split

The dataset was divided into:

* 80% Training data
* 20% Testing data

This allows the model to learn from one portion of the data and be evaluated on unseen data.

### 5. Model Training

A Decision Tree Classifier was used to train the model. Decision Trees are effective for classification tasks and can handle both numerical and categorical data.

### 6. Model Prediction

After training, the model was used to predict the grade class of students from the test dataset.

### 7. Model Evaluation

The model performance was evaluated using accuracy score.

Final Model Accuracy:

92.48%

This indicates that the model correctly predicts student performance in approximately 92% of cases.

---

## Results

The trained machine learning model successfully predicts student academic performance based on the given features.

The results demonstrate that factors such as study time, parental support, extracurricular participation, and GPA significantly influence student performance.

---

## Applications

This project can be useful for:

* Educational institutions to identify students at risk of poor academic performance
* Teachers to provide personalized support to students
* Academic institutions to analyze learning patterns
* Education researchers studying student success factors

---

## Future Improvements

Possible future enhancements include:

* Testing additional machine learning algorithms such as Random Forest and Support Vector Machines
* Implementing feature engineering techniques
* Creating data visualization dashboards
* Deploying the model as a web application

## Author
Shravan Tekam
This project was developed as part of a machine learning learning practice to understand predictive analytics and model development using Python.

---

## License

This project is open-source and available for educational and learning purposes.

