# Student Dropout Analysis
![Python](https://img.shields.io/badge/Python-3.10-blue)  
![License](https://img.shields.io/badge/license-MIT-green)

---

## Project Overview
This project analyzes factors influencing student dropout rates and ships a reproducible modeling pipeline so you can quantify risk, audit fairness, and experiment with interventions.

## Data Source
- **Source**: Survey from 40,000+ students across multiple European institutions.

- **Key Features Used**:
    - `Gender`: indicating the gender of the student (Male/Female).
    - `Student's Performance`: indicating the student's academic performance (e.g., GPA).
    - `Class time`: indicating the student's class time (e.g., Morning/Evening).
    - `Output`: indicating the student's status (e.g., Dropout, Enrolled, Graduate).

## Key Findings
1. **Performance dominates**: Academic performance features (grades/approvals) drive the largest lift in predictive power for dropout.
2. **Class time signal**: Evening attendance is associated with higher dropout probability even after controlling for performance.
3. **Fairness gap is small but monitored**: Precision/recall for the dropout class is comparable across gender and class-time groups (see audit below).

## Recommendations
1. **Targeted Support**: Implement targeted support for male students and those with early performance struggles.
2. **Academic Interventions**: Offer proactive tutoring and milestone checks during the first two semesters.
3. **Flexible Class Scheduling**: Provide flexible scheduling options for students attending evening classes.

## Modeling Pipeline (resume-ready highlights)
- Built end-to-end pipeline with imputation, outlier clipping, one-hot encoding, and standardized numeric features.
- Benchmarked baseline vs tuned models; grid-searched random forest is currently best.
- Added fairness audit for gender and class-time groups, focused on dropout detection precision/recall.

### Repro steps
```bash
make install      # optional: set up .venv
make train        # trains models, saves reports/model_report.json
make test         # runs lightweight feature-engineering tests
```

### Latest metrics (held-out 20% test set)
- Baseline dummy: accuracy 0.41, F1-macro 0.36, ROC-AUC(ovr) 0.52
- Logistic regression: accuracy 0.74, F1-macro 0.70, ROC-AUC(ovr) 0.87
- Tuned random forest (best): accuracy 0.75, F1-macro 0.71, ROC-AUC(ovr) 0.89  
  Fairness (dropout precision/recall):  
  - Gender — Female precision 0.80 / recall 0.71; Male precision 0.85 / recall 0.68  
  - Class time — Day precision 0.83 / recall 0.67; Evening precision 0.83 / recall 0.83

## Visualizations
1. **Student Distribution by Nationality**

    <img src="pictures/student_distribution_by_nationality.png" alt="Student Nationality Distribution" width="600">
    
    *Portuguese students make up the majority, followed by English and Italian students, showing a strong concentration of specific nationalities.*
2. **Average Performance Score by Output**
    
    <img src="pictures/avg_performance_score_by_output.png" alt="Average Performance Score" width="600">
   
    *Graduates have the highest average performance score, while dropouts score significantly lower.*
3. **Student Output by Class Time**
    
    <img src="pictures/student_output_by_class_time.png" alt="Student Output by Class Time" width="600">
    
    *Daytime classes have more students, but evening classes have a relatively higher dropout proportion.*
4. **Average Dropout by Gender**
    
    <img src="pictures/avg_dropout_by_gender.png" alt="Average Dropout by Gender" width="600">
   
    *Male students have a higher average dropout rate compared to female students.*

## Conclusion
This analysis provides valuable insights into the factors contributing to student dropout rates. By understanding these factors, educational institutions can implement targeted strategies to reduce dropout rates and improve student retention. The findings highlight the importance of addressing gender disparities, supporting at-risk students, and considering the impact of class schedules on student success. For further planning, we will conduct a more detailed analysis on the other factors that may influence dropout rates, such as socioeconomic status, family background, and mental health support then develop a machine learning model to early predict dropout risk based on these factors thereby allowing for proactive interventions.
