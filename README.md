# Salary Prediction Project

![Snip20210804_1](https://user-images.githubusercontent.com/24769002/128248342-c9d1b353-7cac-49bf-aa4c-59c6e11ebce3.png)

**Purpose**
------------------------
Perhaps one of the most important factors in hiring and job retention is an employee's salary. Companies that compete within the same sector should always have an indicator of how much they should compensate for their own job openings. This in return allows them to retain their employees and compete for talent not only within their own sectors but other deifferent sectors as well. This project aims to examine a set of job postings with their corresponding salaries and predict salaries for a new set of job postings. As an example salary prediction project, this would be useful in arbitrating the salaries with the use of machine learning models that would predict those values.

**Datasets**
------------------------
The following data sets were provided:

* **train_features.csv** - This file represents a total of 1,000,000 rows with 8 columns (header not included) where each row examines a unique job id along with list of attributes relating to that unique job id.

* **train_salaries.csv** - This file represent a total of 1,000,000 rows with 2 columns (header not included) where each row represents a unique job id along with its corresponfing salary. This file along with train_features.csv will be used for machine learning models.  

* **test_features.csv** - Identical to train_features.csv, where each row examines a unique job id along with list of attributes relating to that unique job. This file will be used to predict salaries.

**Feature Definitions**
------------------------
* **jobId** - Primary key which identifies a distinct job
* **companyId** - Unique Id that indentifies a company
* **jobType** - Defines the level of position a such as CEO, CFO CTO, JANITOR, JUNIOR, MANAGER, SENIOR, VP
* **degree** - Describes the level education such as NONE, HIGH SCHOOL, BACHELORS, MASTERS, DOCTORAL diploma
* **major** - Represents a specific level of specialization at a college or university
* **industry** - Characterizes a specific sector or industry such as OIL, FINANCE, EDUCATION, HEALTH, etc
* **yearsExperience** - Specifies the required number of years for the indicated/listed job 
* **milesFromMetropolis** - Designates the job distance from a major city in miles

**Feature Summary**
------------------------
### Distribution Plots
![Snip20210805_23](https://user-images.githubusercontent.com/24769002/128412053-c2feb56c-aaf9-4efa-9beb-e47c0fa6ff8c.png)

Upon checking for missing values, duplicated values, and values where salary is greater than or equal to $0, it can be noted that the average salary is whithin the range of the middle salary. However, due to some exceptions where certain jobs are being paid more than $220 (see graph above), it can be concluded that average salaries within our data will be greater than the median. This means that by looking at average salary (i.e. mean), the data itself will give a better indication on how salaries are allocated. This will allow us to compare average salaries to other features.

### Average Salary vs. Job Type
![Snip20210805_10](https://user-images.githubusercontent.com/24769002/128395976-a52c155f-3767-41f9-8a96-63ff44d3784c.png)

Comparing the average salary by job type shows that the c-suite postions end up making the most.

### Average Salary vs. Degree
![Snip20210805_12](https://user-images.githubusercontent.com/24769002/128395990-de7a2af1-693e-442b-b636-cfaf888ab4e8.png)

Those who have at least bachelors degree or higher tend to have a higher than average salary.

### Average Salary vs. Major
![Snip20210805_13](https://user-images.githubusercontent.com/24769002/128396007-fe585bd9-ffa8-4fb9-9053-0b5f7c3e4a4e.png)

Business and Engineering majors tend to make more in average salaries in comparison to other majors.

### Average Salary vs. Industry
![Snip20210805_15](https://user-images.githubusercontent.com/24769002/128396023-181e3026-0330-4a12-8c75-af4f25841bb6.png)

The Finance and Oil Industry have higher average salaries than Education, Service, Auto, Health, and Web.


Overall, average salary as a metric provides a more accurate picture when comparing average salary amongst other features. Other things that were observed for comparison were as follows:
* There is a postive relationship or connection with having a higher salary and more years of experience.
* There is also a negative relationship or connection with having a higher salary and the distance a job is located from a major city.

### Correlation Matrix
![Snip20210805_19](https://user-images.githubusercontent.com/24769002/128408799-12dc3c9c-8178-493c-b6d2-2435c802093f.png)

In summary, based on the above matrix we can observe the following:
* There is a postive relationship with salary and jobType, degree, major, industry, yearsExperience
* There is a negative relationship with salary and milesFromMetropolis

**Regression Models**
------------------------
The following models were used to asses the prediction of our salaries for the following reasons:
  * Linear Regression: Basic regression model which can be used for any data set and size.
  * Random Forest: A low bias model that is very fast and powerful to solve regression/classificiation problems.
  * Gradient Boosting: A fast and high performanced model that can create simple individual models by combining them into a new one.

The Mean Squared Error is used to check how close our estimates are to the actual values. In reference to predciting salaries, the lower the MSE, the better our prediction. A 0 MSE means that the model is perfect.

|Models|Mean Squared Error|
|---|---|
|Linear Regression|385.698613|
|Random Forest|383.789228|
|Gradient Boosting Regressor|361.850797|

Based on the above models, Gradient Boosting Regressor preformed better in terms of predicting salaries for the given job ids. 

|jobId|predicted_salary|
|---|---|
|JOB1362685407687|143.293156|
|JOB1362685407688|140.000311|
|JOB1362685407689|136.299671|
|JOB1362685407690|125.276273|
|JOB1362685407691|116.435501|

Based on implementing GBR on our test_features.csv, we see that Gradient Boosting was able to predict the salaries for the first 5 Jobs.

**Feature Importance**
------------------------
![Snip20210805_25](https://user-images.githubusercontent.com/24769002/128416275-07a10310-41bc-4ee8-94a1-d69446f3248c.png)

Upon extracting a feature importance, we can see that years of experience and miles from metropolis are two of the most important features in predicting salaries. Why is this important? Knowing this information allows to adjust our features so that we may inprove our models.

**Closing Remarks**
------------------------
Understanding how much a job should pay would be better useful if allow prediction models to asses those salaries. This in return would reduce biases in the workplace and allow companies/organizations to better compete for talent. We know the Gradient Boosting Regressor is better model to predict salaries. How can we improve this model? We need to adjust our features and adjust our models. Also exploring other models as well would be useful in prediction. For simplicity's sake, we decided to choose only 3 models.

