# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### NGUYEN DUC HUY

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
After conducting the following 03 experiments:

- Initial Raw Submission 
- Added Features Submission (EDA + Feature Engineering) 
- Hyperparameter Optimization (HPO)

I observed that all the predicted values were consistently greater than or equal to zero. In order to finalize my submission, I decided to modify any values less than zero and set them equal to zero.

### What was the top ranked model that performed?
The top-ranked model was the (add features) model named WeightedEnsemble_L2, with a validation RMSE score of 35.9 and the best Kaggle score of 0.572 (on test dataset)

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
The "datetime" feature was parsed as a datetime variable to extract the hour information from the timestamp. The independent features "season" and "weather" were initially read as integers. However, since these variables represent categorical information, they were transformed into categorical data types.

### How much better did your model preform after adding additional features and why do you think that is?
Upon adding more features, the MSE score has noticeably changed from 1.791 to 0.674. This suggests that the "datetime" column may have an impact on the training model's performance.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
After conducting multiple training iterations and tuning hyperparameters on various models such as XT (Extra Trees), GBM (Gradient Boosting Machine), and RF (Random Forest), it has been observed that the results have improved compared to the previous training runs.

### If you were given more time with this dataset, where do you think you would spend more time?
If given more time to work with the dataset, you can explore additional potential outcomes by running AutoGluon for an extended period using a high-quality preset and enhanced hyperparameter tuning. This can involve allowing the algorithm more time to optimize and fine-tune the models, leading to potentially improved results. 

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|eval_metric=root_mean_squared_error	|auto_stack=False	|presets=[best_quality]	|1.791|
|add_features|eval_metric=root_mean_squared_error	|auto_stack=False	|presets=[best_quality]	|0.674|
|hpo|eval_metric=root_mean_squared_error	|auto_stack=False	|presets=[optimize_for_deployment]	|0.572|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](img/model_test_score.png)

## Summary
In the project of predicting bike sharing, I have gained experience in using Sagemaker and AutoGluon for data processing, visualization, and model training. During this process, I observed that hyperparameter tuning with AutoGluon led to improved performance compared to the initial raw submission. This indicates that fine-tuning the hyperparameters of the models using AutoGluon's capabilities has positively impacted the predictive accuracy of the bike sharing model.