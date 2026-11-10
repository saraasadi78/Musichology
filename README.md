# Music and Mental Health Recognition using Machine Learning

This machine-learning project explores the intriguing relationship between individuals' music listening habits and their mental well-being.
The primary objective is to predict mental health indicators, including Obsessive-Compulsive Disorder (OCD), Anxiety, Depression, and Insomnia, based on an individual's music taste and listening habits.

Description:

Conducted comprehensive data preprocessing on a large dataset containing demographic information, music genre preferences, daily activity hours, and mental health rankings (OCD, insomnia, depression, anxiety).

Employed data visualization techniques to gain insights into the relationships between music preferences, daily activity patterns, and mental health outcomes, aiding in feature selection and model interpretation.

Utilized  Neural network and linear regression algorithms to build a predictive model for mental health rankings based on input features.

Achieved a high-performing predictive model with an emphasis on interpretability, enabling stakeholders to understand the key factors influencing mental health outcomes.

Implemented the model into a practical tool for real-time predictions, enhancing the project's practicality and usability.

Demonstrated proficiency in data science methodologies, including data preprocessing, exploratory data analysis, machine learning, and model deployment, contributing to the project's success.

## Key findings

- The tuned Random Forest was the strongest overall model in five-fold cross-validation, but predictive signal was limited: R² was **0.04** for Anxiety, **0.07** for Depression, **-0.02** for Insomnia, and **-0.00** for OCD.
- Cross-validated RMSE was **2.73** for Anxiety, **2.91** for Depression, **3.08** for Insomnia, and **2.83** for OCD on the 0–10 self-reported scales.
- SHAP explanations identified age, hours listened per day, BPM, and genre-frequency indicators as the most influential features. The strongest features differed by target, with age prominent for Anxiety and Depression and listening hours prominent for Insomnia and OCD.
- These are predictive associations in a self-reported survey, not evidence that music habits cause changes in mental health.
