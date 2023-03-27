# Anomaly Based Network Intrusion Detection Module based on Support Vector Machine Algorithm

To accomplish its goals, this project will rely on the publicly available CSE-CIC-IDS 2018 dataset. This dataset contains a significant amount of network traffic data captured from a realistic environment, including labeled examples of both normal and anomalous network behavior. By utilizing this dataset, the project can test the effectiveness of its anomaly detection algorithms and refine them as necessary.

The main focus of this project is on the SVM algorithm, which will be used to analyze and visualize the network traffic data. By leveraging the power of AI, the SVM algorithm can identify patterns and anomalies in the data that may be difficult for humans to detect. This approach offers an efficient and effective method for detecting network anomalies, which is essential for securing critical network infrastructures against potential attacks.

In conclusion, this project aims to develop an innovative anomaly detection tool that utilizes the SVM algorithm and AI technology to detect and flag network anomalies. By providing a more intuitive and user-friendly interface for analyzing the results, the tool can be used to secure critical network infrastructures against potential attacks. Ultimately, the success of this project could have far-reaching implications for the field of network security and the protection of vital network infrastructures.
<br><br>

## Technologies Used⚙️<hr>
<img align=right width=200px src="https://octodex.github.com/images/daftpunktocat-guy.gif">
This project was built using the following technologies:

<ui>
<li> Jupyter notebook 📓
<li> python 🐍
<li> sklearn libraries 🧮
<li> seaborn 🌊
<li> pandas 🐼
<li> matplotlib.pyplot 📈
</ui>
<br>
<br>



## Overview setup📋<hr>
## <u>Data manipulation </u>

<p align="center" >
<img width=50% src="https://user-images.githubusercontent.com/78523790/227912704-139e0816-6bdb-4726-84dd-ff9f26f881f9.png"> <br>
<i>Data relation between Benign and Malignant </i>
</p>

To clean and prepare the dataset for training and testing, I first load it into the df variable. Then, I replace spaces with underscores and filter rows related to HTTP and HTTPS traffic. Next, I check for null and infinite values and remove them if any exist. I also check the data type of each column and remove the majority of object columns.

I change the label data as follows: benign is changed to 0 and DDoS attacks is changed to 1. The original dataset had 2089686 rows with only 576191 of them being labeled as 1. I reduce the number of 0 rows to match the number of 1 rows.

Next, I create a duplicate dataset with 2000 rows from the previous dataset for training and testing purposes. I convert the dataset to NumPy arrays and test for any infinite or infinite values and replace them if they exist.

Finally, I divide the data into training and testing sets, allocating 30% of the data for testing, and set the random seed to 42. This process ensures that the dataset is suitable for training and testing my model.

<br>

## <u> SVM training setup </u>

For training the SVM model, I used the GridSearchCV function to select the best parameters for training. I chose the rbf kernel and set the parameter grid to the following values: 'C': [0.1, 1, 10, 100] and 'gamma': ['scale', 'auto', 0.1, 1, 10]. I set the number of cross-validation folds to 50 (cv=50), used all available processors for parallel computation (n_jobs=-1), and set the verbosity level to 1 (verbose=1).

This setup allowed me to efficiently train the SVM model and select the best hyperparameters for optimal performance.
<br><br>

## Results 🎉<hr>

## <u> Grid-Search results </u>

<table align="center" style="border-collapse: collapse;">
  <tr>
    <td style="border: none; width:50%;"><p align="center"><img width=100% src="https://user-images.githubusercontent.com/78523790/227913062-9cab5829-bba4-4c11-b2f9-80eb9acd7517.png"><br><i>Grid-Search Score</i></p></td>
    <td style="border: none;"><p align="left">Based on the results of the SVM model training using GridSearchCV, it appears that the best kernel for this dataset is rbf, with the best C value being 100 and the best gamma value being scale.<br><br>The model also achieved high accuracy, with an accuracy score of 0.955. Additionally, the precision score of 0.9249146757679181 indicates that the model had a low rate of false positives, while the recall score of 0.9818840579710145 suggests that the model was effective at identifying true positives. The F1 score of 0.952548330404218 also indicates a good balance between precision and recall.<br><br>Overall, these results demonstrate that the SVM model trained on this dataset using GridSearchCV is effective at detecting network intrusions with high accuracy and minimal false positive
</p></td>
  </tr>
</table>

## <u> Confusion Matrix </u>

<table align="center" style="border-collapse: collapse;">
  <tr>
    <td style="border: none; width:50%;"><p align="center"><img width=100% src="https://user-images.githubusercontent.com/78523790/227913195-e98e9580-44d3-4fab-a79e-bcc3ada983d0.png"><br><i>Confusion Matrix results</i></p></td>
    <td style="border: none; "><p align="left">The confusion matrix provided shows that the model made 302 true positive predictions and 271 true negative predictions. It also made 22 false positive predictions and 5 false negative predictions. <br><br>Using this information, we can calculate various performance metrics such as precision, recall, and F1-score. For example, precision can be calculated as the ratio of true positive predictions to the sum of true positive and false positive predictions. In this case, the precision score would be 0.932. Recall can be calculated as the ratio of true positive predictions to the sum of true positive and false negative predictions, which in this case would be 0.982. The F1-score can be calculated as the harmonic mean of precision and recall, resulting in a score of 0.956. <br><br>Overall, the confusion matrix can provide valuable insights into the performance of the model and can be used to calculate various performance metrics that can help to evaluate the effectiveness of the model at detecting network intrusions.</p></td>
  </tr>
</table>


## <u> Classification report </u>

<p align="center" >
<img width=80% src="https://user-images.githubusercontent.com/78523790/227913319-2691ad0b-178f-48ea-9f5a-8bb12d0ec5db.png"><br>
<i>Classification report results</i></p>

The classification report provides a summary of various performance metrics of the model, such as precision, recall, and F1-score, for both the benign traffic (0.0) and the network intrusions (1.0) classes.

For the benign traffic class (0.0), the precision score is 0.98, the recall score is 0.93, and the F1-score is 0.96. This means that the model correctly identified 93% of the benign traffic instances, and out of all the instances that were classified as benign, 98% were actually benign.

For the network intrusions class (1.0), the precision score is 0.92, the recall score is 0.98, and the F1-score is 0.95. This means that the model correctly identified 98% of the network intrusion instances, and out of all the instances that were classified as network intrusions, 92% were actually network intrusions.

The accuracy of the model is 0.95, which means that it correctly classified 95% of the instances in the dataset.

The macro average of precision, recall, and F1-score is 0.95, which is the average of the scores for both classes. The weighted average is 0.96, which takes into account the number of instances in each class.

Overall, the classification report provides a comprehensive summary of the performance of the model, and can be used to evaluate its effectiveness in identifying network intrusions.

<br>

##  <u> ROC curve, Precision-Recall curve and F1-score </u> 

<table style="border: none; margin: auto;">
  <tr style="border: none;">
    <td style="border: none; text-align: center;">
      <img width="100%" src="https://user-images.githubusercontent.com/78523790/227913534-97e591b5-86c8-459e-9ddb-6d00465bfd47.png">
    </td>
    <td style="border: none; text-align: center;">
      <img width="100%" src="https://user-images.githubusercontent.com/78523790/227913640-bed4e4b2-6850-4f24-84b8-8a24f68bce85.png">
    </td>
    <td style="border: none; text-align: center;">
      <img width="100%" src="https://user-images.githubusercontent.com/78523790/227913712-546dc0a0-3b45-459f-91ab-7900caa307db.png">
    </td>
  </tr>
  <tr style="border: none;">
    <td style="border: none; text-align: center;"><i>ROC Curve</i></td>
    <td style="border: none; text-align: center;"><i>Precision Recall curve</i></td>
    <td style="border: none; text-align: center;"><i>F1 score</i></td>
  </tr>
</table>
<br>

The SVM model was trained using the Radial Basis Function (RBF) kernel with the regularization parameter (C) set to 100 and the gamma parameter set to 'scale'. The model was trained using gridserchcv to select the best hyperparameters, where C was chosen from [0.1, 1, 10, 100] and gamma was chosen from ['scale', 'auto', 0.1, 1, 10]. The cross-validation was set to 50 and n_jobs=-1 with verbose=1.

The confusion matrix shows that there were 302 true negative (TN) cases and 271 true positive (TP) cases in the dataset, with 22 false negative (FN) cases and 5 false positive (FP) cases. The F1 score of the SVM model on the test set is 0.95, indicating a high level of performance in correctly predicting the target variable.

The ROC curve shows that the SVM model has a high TPR, meaning it is able to correctly identify a high proportion of positive instances in the dataset. However, the FPR values are not negligible, indicating that the model may have some difficulty in correctly classifying negative instances.

The model achieved an accuracy score of 0.955, indicating that it was able to classify 95.5% of the test data correctly. The precision and recall scores were also high, with values of 0.9249 and 0.9819 respectively. This suggests that the model was able to accurately identify both positive and negative cases in the test data.

The F1 score was also used to evaluate the impact of the hyperparameter C on the model's performance. For a small value of C (0.1), the model has a lower F1 score of 0.74, indicating that the model's performance is poorer when it is underfitting. As the value of C increases to 1, the F1 score improves to 0.76, indicating better performance. The F1 score continues to increase as C increases, with the highest F1 score of 0.95 achieved at C = 100. This suggests that the model's performance continues to improve as it becomes more complex, with a high level of regularization at small C values resulting in poor performance.

Overall, the SVM model using RBF kernel with C=100 and gamma=scale appears to be a good model for this dataset, with high accuracy, precision, recall, and F1 score. However, it is important to note that the results may not generalize to other datasets, and further testing and validation may be necessary.

<br>

## Conclusion 🔍✅🎉<hr>

Based on the results of the SVM model trained on the dataset, it can be concluded that the RBF kernel with hyperparameters C=100 and gamma=scale is an effective choice for this problem. The model achieved an accuracy score of 0.96 on the test data, with high precision and recall scores. The confusion matrix also shows that the model was able to accurately classify most of the test data.

The ROC curve suggests that the model has a high true positive rate, indicating that it is able to correctly identify a high proportion of positive instances in the dataset. However, the false positive rate values are not negligible, suggesting that the model may have some difficulty in correctly classifying negative instances.

The F1 score of the model on the test set is 0.95, indicating a high level of performance in correctly predicting the target variable. The impact of the hyperparameter C on the model's F1 score was also evaluated, and it was found that increasing the value of C improved the model's performance.

Overall, the SVM model with RBF kernel and hyperparameters C=100 and gamma=scale appears to be a good choice for this dataset. However, it is important to note that the results may not generalize to other datasets, and further testing and validation may be necessary.

<br>

## Credits 👏 <hr>

This project was developed by [Your Name] as a personal project without a supervisor. The following Python libraries were used for data processing, visualization, and machine learning:

pandas (v1.3.4) - Data manipulation and analysis
glob - File path management
numpy (v1.21.4) - Numerical computing
seaborn (v0.11.2) - Data visualization
matplotlib (v3.4.3) - Plotting and visualization
scikit-learn (v1.0) - Machine learning framework
The dataset used in this project is the CSE-CIC-IDS 2018 dataset, which was obtained from Kaggle. The dataset contains network traffic data that has been labeled as either normal or malicious. The dataset was preprocessed using the Python libraries mentioned above.

The machine learning model used in this project is the Support Vector Machine (SVM) algorithm with Radial Basis Function (RBF) kernel. The SVM model was trained and evaluated using scikit-learn. The performance of the model was evaluated using various metrics such as accuracy, precision, recall, F1 score, and ROC curve.

Overall, this project serves as an example of using machine learning algorithms to classify network traffic as either normal or malicious. The insights gained from this project can be used to improve network security and prevent cyber attacks.

<br>

## License 📄 <hr>
This project is licensed under the MIT License.
