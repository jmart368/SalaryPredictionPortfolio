# Salary Prediction Project

![Snip20210804_1](https://user-images.githubusercontent.com/24769002/128248342-c9d1b353-7cac-49bf-aa4c-59c6e11ebce3.png)

**Purpose**
------------------------
Perhaps one of the most important factors in hiring and job retention is an employee's salary. Companies that compete within the same sector should always have an indicator of how much they should compensate for their own job openings. This in return allows them to retain their employees and compete for talent. This project aims to examine a set of job postings with their corresponding salaries and predict salaries for a new set of job postings.

**Datasets**
------------------------
The following data sets were provided:

* **train_features.csv** - This file represents a total of 1,000,000 rows with 8 columns (header not included) where each row examines a unique job id along with list of attributes relating to that unique job id.

* **train_salaries.csv** - This file represent a total of 1,000,000 rows with 2 columns (header not included) where each row represents a unique job id along with its corresponfing salary. This file along with train_features.csv will be used for machine learning models.  

* **test_features.csv** - Identical to train_features.csv, where each row examines a unique job id along with list of attributes relating to that unique job. This file will be used to predict salaries

**Feature Definitions**
------------------------
* **jobId** - Primary key which identifies a distinct job
* **companyId** - Unique Id that indentifies a company
* **jobType** - Defines the level of position a such as CEO, CFO CTO, JANITOR, JUNIOR, MANAGER, SENIOR, VP
* **degree** - Describes the level education from NONE up to a DOCTORAL degree
* **major** - Represents a specific level of specialization at a college or university
* **industry** - Characterizes a specific sector or industry 
* **yearsExperience** - Specifies the required number of years for a job 
* **milesFromMetropolis** - Designates the distance from a major city in miles

**Feature Summary**
------------------------
### Distribution Plots
![Snip20210805_23](https://user-images.githubusercontent.com/24769002/128404699-f61843cd-c034-4921-95ed-f401a377e478.png)

Upon checking for missing values, duplicated values, and values where salary is greater than or equal to $0, it can be noted that the average salary is whithin the range of middle salary. However, due to some exceptions where certain jobs are being paid more than $220, it can be concluded that average salaries within our data will be greater than the median. This means that by looking at average salary (i.e. mean), the data itself will give a better indication on how salaries are allocated in comparison to other features.  

### Salary vs. Job Type
![Snip20210805_10](https://user-images.githubusercontent.com/24769002/128395976-a52c155f-3767-41f9-8a96-63ff44d3784c.png)


### Salary vs Degree
![Snip20210805_12](https://user-images.githubusercontent.com/24769002/128395990-de7a2af1-693e-442b-b636-cfaf888ab4e8.png)

### Salary vs Major
![Snip20210805_13](https://user-images.githubusercontent.com/24769002/128396007-fe585bd9-ffa8-4fb9-9053-0b5f7c3e4a4e.png)

### Salary vs Industry
![Snip20210805_15](https://user-images.githubusercontent.com/24769002/128396023-181e3026-0330-4a12-8c75-af4f25841bb6.png)



**Feature Engineering**
------------------------


**Regression Models**
------------------------

**Feature Importance**
------------------------
![Snip20210805_24](https://user-images.githubusercontent.com/24769002/128404804-e8ec4c8d-3c04-462f-9787-f97649aec4a9.png)

**Conclusion**
------------------------
