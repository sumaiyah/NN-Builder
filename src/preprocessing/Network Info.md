# Artif-1

```pseudocode
x1 {0, 0.5, 1}
x2 {0, 0.25, 0.5, 0.75, 1}
x3 {0,1}
x4 {0,1}
x5 {0,1}

IF x1 = x2 THEN y = y1
IF x1 > x2 AND x3 > 0.4 THEN y = y1
IF x3 > x4 AND x4 > x5 AND x2 > 0 THEN y = y1 ELSE y = y0
```

**Train**: 24000

**Test**: 6000

**Network **: 5-10-5-2

# Artif-2

# BreastCancer

src: : [https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

**30** input features

Class distribution: 357 benign, 212 malignant: 569 in total 

# LetterRecognition

src: https://archive.ics.uci.edu/ml/datasets/Letter+Recognition

Original pixel graphics have been transformed to a number of 16 attributes representing special characteristics, 0-15 indicate each attribute

Reducing the dataset to perform binary classification task
- class_0: 'A'
- class_1: any other letter

# MNIST

# Results

By default networks trained with:

- ```
  batch_size=25,
  epochs=200
  ```



Decompositional (DeepRED) Rule Extraction

| Dataset            | NN Structure  | Train | Test | NN Accuracy | NN Train Time (s) | Rules Extracted | Rule Accuracy | Rule Fidelity | Av Number Terms per rule | Rule Extraction Time (s) | Memory Usage (Mb) |
| ------------------ | ------------- | ----- | ---- | ----------- | ----------------- | --------------- | ------------- | ------------- | ------------------------ | ------------------------ | ----------------- |
| Artif-1            | 5-10-5-2      | 24000 | 6000 | 100         | 228.87            | 4, 6            | 100           | 100           | 2.5                      | 11.57                    | 92.03             |
| Artif-2            | 5-10-5-2      | 4000  | 1000 | 100         | 44.7              | 16, 42          | 94.2          | 94.2          | 4.60                     | 6.10                     | 47.55             |
| BreastCancer       | 30-16-2-2     | 455   | 114  | 97.37       | 5.107             | 2, 2            | 89.47         | 90.35         | 1.5                      | 0.665                    | 9.93              |
| LetterRecognition* | 16-40-30-2    | 1262  | 316  | 97.15       | 16.10             |                 |               |               |                          |                          |                   |
| MNIST*             | 784-10-5-2    | 11824 | 2956 | 99.86       | 230.9             | 9, 6            | 99.996        | 99.76         | 2.53                     | 126.1                    | 1139.57           |
| MB-ER              | 1000-100-50-2 | 1584  | 396  | 93.43       | 33.97             | 168, 40         | 94.7          | 94.2          | 5.55                     | 120.85                   | 311.53            |

*Working with binarized data (transform problem into a binary classification task)



Pedagogical Baseline

| Dataset           | Rules Extracted | Rule Accuracy | Rule Fidelity | Av Number Terms per rule | Rule Extraction Time (s) | Memory Usage (Mb) |
| ----------------- | --------------- | ------------- | ------------- | ------------------------ | ------------------------ | ----------------- |
| Artif-1           | 5, 5            | 100           | 100           | 2.6                      | 0.853                    | 25.33             |
| Artif-2           | 6, 4            | 100           | 100           | 2.8                      | 0.25                     | 0.74              |
| BreastCancer      | 5, 6            | 93.86         | 96.49         | 2.27                     | 0.24                     | 1.11              |
| LetterRecognition | 10, 6           | 95.57         | 93.4          | 3                        | 0.55                     | 2.09              |
| MNIST             | 8, 6            | 99.696        | 99.763        | 2.57                     | 32.2                     | 575.6             |
| MB-ER             | 10, 9           | 91.41         | 92.42         | 3.05                     | 13.69                    | 204.8             |

----

```
MNIST with [784, 20, 10, 2]

--- Rule Extraction took 132.1210401058197 seconds and used 1202.93359375 Mb to execute ---
Accuracy:  0.9979702300405954
Fidelity:  0.9983085250338295
Comprehensibility:  
    Class:  Zero | n_rules: 9 | number of terms: min: 1 max: 3 average: 2.000000 | 
    Class:  One | n_rules: 5 | number of terms: min: 1 max: 4 average: 3.200000 | 
  Overall average n_terms per rule:  2.4285714285714284
------------------------------------------------------------------------------------------------------
MNIST with [784, 30, 20, 10, 2]

--- Rule Extraction took 144.0135440826416 seconds and used 1299.97265625 Mb to execute ---
Accuracy:  0.996617050067659
Fidelity:  0.9969553450608931
Comprehensibility:  
    Class:  Zero | n_rules: 6 | number of terms: min: 2 max: 3 average: 2.333333 | 
    Class:  One | n_rules: 4 | number of terms: min: 2 max: 3 average: 2.500000 | 
  Overall average n_terms per rule:  2.4
```

