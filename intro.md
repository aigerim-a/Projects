# Welcome to your Jupyter Book

This is a small sample book to give you a feel for how book content is
structured.
It shows off a few of the major file types, as well as some sample content.
It does not go in-depth into any particular topic - check out [the Jupyter Book documentation](https://jupyterbook.org) for more information.

Check out the content pages bundled with this sample book to see more.

```{tableofcontents}
```


# Catboost


# What Catboost can do?
    1. Classification
    2. Regression
    3. Multiclassification
    4. Ranging
    5. Metrics
    6. etc.
    
# Operating Principle
## Decision Tree

The operating algorithm is as follows: for each document, there is a set of feature values, and there is a tree with conditions at its nodes. If the condition is met, the algorithm moves to the right child of the node; otherwise, it goes to the left. One needs to traverse the tree to a leaf according to the feature values for the document. The value of the leaf corresponds to the output for each document. That is the answer.

Boosting

The idea behind the boosting approach is to combine weak (with low generalization ability) functions built during an iterative process, where at each step, a new model is trained using data on the errors of the previous ones. The resulting function is a linear combination of basic, weak models.

Next, boosting decision trees will be considered. Several trees will be built, and adding new trees should reduce the error. In total, with a sufficiently large number of trees, the error can be significantly reduced. However, it is essential to remember that the more trees, the longer the model takes to train, and at some point, the quality improvement becomes insignificant.
Gradient Boosting

## CatBoost is based on gradient boosting.

The gradient of the error function includes all derivatives with respect to all values of the function.

Gradient boosting is a machine learning method that creates a predictive model in the form of an ensemble of weak prediction models, usually decision trees. It builds the model step by step, allowing the optimization of any differentiable loss function.

## Gradient boosting

Boosting is a method which builds a prediction model $F^{T}$ as an ensemble of weak learners $F^{T} = \sum\limits_{t=1}^{T} f^{t}$.

In our case, $f^{t}$ is a decision tree. Trees are built sequentially and each next tree is built to approximate negative gradients $g_{i}$ of the loss function $l$ at predictions of the current ensemble:
$g_{i} = -\frac{\partial l(a, y_{i})}{\partial a} \Bigr|_{a = F^{T-1}(x_{i})}$
Thus, it performs a gradient descent optimization of the function $L$. The quality of the gradient approximation is measured by a score function $Score(a, g) = S(a, g)$.

# Features of CatBoost
## Operating Modes

- Regression
- Classification

#### Loss Function: Maximizes the probability that all objects in the training set are classified correctly, where probability is the sigmoid function applied to the formula's value.

#### predict_proba Function: Outputs ready probabilities. It's important to note that these probabilities cannot be summed.

#### predict Function: Outputs raw results. Such results can be combined, for example, with the results of other models.

- Multiclass Classification

- Ranking

## Metrics

CatBoost supports a variety of metrics, such as:

    Regression: MAE, MAPE, RMSE, SMAPE, etc.
    Classification: Logloss, Precision, Recall, F1, CrossEntropy, BalancedAccuracy, etc.
    Multiclass Classification: MultiClass, MultiClassOneVsAll, HammingLoss, F1, etc.
    Ranking: NDCG, PrecisionAt, RecallAt, PFound, PairLogit, etc.


<span style="display:none" id="q_demo_seq">W3sicXVlc3Rpb24iOiAiXHUwNDFmXHUwNDQwXHUwNDNlXHUwNDM0XHUwNDNlXHUwNDNiXHUwNDM2XHUwNDM4XHUwNDQyXHUwNDM1IFx1MDQzZlx1MDQzZVx1MDQ0MVx1MDQzYlx1MDQzNVx1MDQzNFx1MDQzZVx1MDQzMlx1MDQzMFx1MDQ0Mlx1MDQzNVx1MDQzYlx1MDQ0Y1x1MDQzZFx1MDQzZVx1MDQ0MVx1MDQ0Mlx1MDQ0YzogJDIkLCAkXFxmcmFjIDk0JCwgJFxcZnJhY3s2NH17Mjd9JCwgPyIsICJ0eXBlIjogIm51bWVyaWMiLCAiYW5zd2VycyI6IFt7InR5cGUiOiAidmFsdWUiLCAidmFsdWUiOiAyLjQ0MTQwNjI1LCAiY29ycmVjdCI6IHRydWUsICJmZWVkYmFjayI6ICJcdTA0MTJcdTA0MzVcdTA0NDBcdTA0M2RcdTA0M2UhIFx1MDQyZFx1MDQ0Mlx1MDQzZSBcdTA0M2ZcdTA0M2VcdTA0NDFcdTA0M2JcdTA0MzVcdTA0MzRcdTA0M2VcdTA0MzJcdTA0MzBcdTA0NDJcdTA0MzVcdTA0M2JcdTA0NGNcdTA0M2RcdTA0M2VcdTA0NDFcdTA0NDJcdTA0NGMgJFxcYmlnKFxcZnJhY3tuKzF9blxcYmlnKV5uJCwgXHUwNDNmXHUwNDQwXHUwNDM4ICRuPTQkIFx1MDQzZlx1MDQzZVx1MDQzYlx1MDQ0M1x1MDQ0N1x1MDQzMFx1MDQzNVx1MDQzYyAkXFxmcmFjezVeNH17NF40fSA9IFxcZnJhY3s2MjV9ezI1Nn0gPSAyLjQ0MTQwNjI1JCJ9LCB7InR5cGUiOiAiZGVmYXVsdCIsICJmZWVkYmFjayI6ICJcdTA0MWVcdTA0NDJcdTA0MzJcdTA0MzVcdTA0NDIgXHUwNDNkXHUwNDM1XHUwNDMyXHUwNDM1XHUwNDQwXHUwNDNkXHUwNDRiXHUwNDM5ISBcdTA0MTdcdTA0MzAgXHUwNDNmXHUwNDNlXHUwNDM0XHUwNDQxXHUwNDNhXHUwNDMwXHUwNDM3XHUwNDNhXHUwNDNlXHUwNDM5IFx1MDQzZlx1MDQ0MFx1MDQzZVx1MDQzYlx1MDQzOFx1MDQ0MVx1MDQ0Mlx1MDQzMFx1MDQzOVx1MDQ0Mlx1MDQzNSBcdTA0M2RcdTA0MzVcdTA0M2NcdTA0M2RcdTA0M2VcdTA0MzNcdTA0M2UgXHUwNDMyXHUwNDRiXHUwNDQ4XHUwNDM1In1dfV0=</span>


<span style="display:none" id="q_deposit_1">W3sicXVlc3Rpb24iOiAiXHUwNDE4XHUwNDQyXHUwNDMwXHUwNDNhLCBcdTA0MzJcdTA0NGIgXHUwNDNmXHUwNDNlXHUwNDNiXHUwNDNlXHUwNDM2XHUwNDM4XHUwNDNiXHUwNDM4ICRcXCQxJCBcdTA0MzIgXHUwNDMxXHUwNDMwXHUwNDNkXHUwNDNhIFx1MDQzZlx1MDQzZVx1MDQzNCAkMTAwXFwlJCBcdTA0MzNcdTA0M2VcdTA0MzRcdTA0M2VcdTA0MzJcdTA0NGJcdTA0NDUuIFx1MDQyMVx1MDQzYVx1MDQzZVx1MDQzYlx1MDQ0Y1x1MDQzYVx1MDQzZSBcdTA0NDMgXHUwNDMyXHUwNDMwXHUwNDQxIFx1MDQzMVx1MDQ0M1x1MDQzNFx1MDQzNVx1MDQ0MiBcdTA0MzRcdTA0MzVcdTA0M2RcdTA0MzVcdTA0MzMgXHUwNDNkXHUwNDMwIFx1MDQzMlx1MDQzYVx1MDQzYlx1MDQzMFx1MDQzNFx1MDQzNSBcdTA0NDdcdTA0MzVcdTA0NDBcdTA0MzVcdTA0MzcgXHUwNDMzXHUwNDNlXHUwNDM0PyIsICJ0eXBlIjogIm51bWVyaWMiLCAiYW5zd2VycyI6IFt7InR5cGUiOiAidmFsdWUiLCAidmFsdWUiOiAyLCAiY29ycmVjdCI6IHRydWUsICJmZWVkYmFjayI6ICJcdTA0MTJcdTA0MzVcdTA0NDBcdTA0M2RcdTA0M2UhICQxMDBcXCUkIFx1MDQzM1x1MDQzZVx1MDQzNFx1MDQzZVx1MDQzMlx1MDQ0Ylx1MDQ0NSBcdTIwMTQgXHUwNDRkXHUwNDQyXHUwNDNlIFx1MDQ0M1x1MDQzNFx1MDQzMlx1MDQzZVx1MDQzNVx1MDQzZFx1MDQzOFx1MDQzNTogJFxcJDEgKyBcXCQxID0gXFwkMiQifSwgeyJ0eXBlIjogInJhbmdlIiwgInJhbmdlIjogWy0xZSsxOCwgMF0sICJjb3JyZWN0IjogZmFsc2UsICJmZWVkYmFjayI6ICJcdTA0MWVcdTA0NDJcdTA0NDBcdTA0MzhcdTA0NDZcdTA0MzBcdTA0NDJcdTA0MzVcdTA0M2JcdTA0NGNcdTA0M2RcdTA0MzBcdTA0NGYgXHUwNDQxXHUwNDQzXHUwNDNjXHUwNDNjXHUwNDMwIFx1MDQzZVx1MDQzN1x1MDQzZFx1MDQzMFx1MDQ0N1x1MDQzMFx1MDQzNVx1MDQ0MiwgXHUwNDQ3XHUwNDQyXHUwNDNlIFx1MDQzMlx1MDQ0YiBcdTA0MzVcdTA0NDlcdTA0NTEgXHUwNDM4IFx1MDQzZVx1MDQ0MVx1MDQ0Mlx1MDQzMFx1MDQzYlx1MDQzOFx1MDQ0MVx1MDQ0YyBcdTA0MzRcdTA0M2VcdTA0M2JcdTA0MzZcdTA0M2RcdTA0NGIuIFx1MDQxZVx1MDQ0N1x1MDQzNVx1MDQzZFx1MDQ0YyBcdTA0NDFcdTA0NDJcdTA0NDBcdTA0NTFcdTA0M2NcdTA0M2RcdTA0NGJcdTA0MzkgXHUwNDMxXHUwNDMwXHUwNDNkXHUwNDNhIFx1MDQzZVx1MDQzYVx1MDQzMFx1MDQzN1x1MDQzMFx1MDQzYlx1MDQ0MVx1MDQ0ZiJ9LCB7InR5cGUiOiAicmFuZ2UiLCAicmFuZ2UiOiBbMCwgMV0sICJjb3JyZWN0IjogZmFsc2UsICJmZWVkYmFjayI6ICJcdTA0MTJcdTA0MzBcdTA0NDggXHUwNDMyXHUwNDNhXHUwNDNiXHUwNDMwXHUwNDM0IFx1MDQ0M1x1MDQzY1x1MDQzNVx1MDQzZFx1MDQ0Y1x1MDQ0OFx1MDQzOFx1MDQzYlx1MDQ0MVx1MDQ0Zj8gXHUwNDEzXHUwNDQwXHUwNDMwXHUwNDMxXHUwNDUxXHUwNDM2IFx1MDQ0MVx1MDQ0MFx1MDQzNVx1MDQzNFx1MDQ0YyBcdTA0MzFcdTA0MzVcdTA0M2JcdTA0MzAgXHUwNDM0XHUwNDNiXHUwNDRmISJ9LCB7InR5cGUiOiAidmFsdWUiLCAidmFsdWUiOiAxLCAiY29ycmVjdCI6IGZhbHNlLCAiZmVlZGJhY2siOiAiXHUwNDEwIFx1MDQ0MVx1MDQzY1x1MDQ0Ylx1MDQ0MVx1MDQzYiBcdTA0M2RcdTA0MzVcdTA0NDFcdTA0NDJcdTA0MzggXHUwNDM0XHUwNDM1XHUwNDNkXHUwNDRjXHUwNDMzXHUwNDM4IFx1MDQzMiBcdTA0MzFcdTA0MzBcdTA0M2RcdTA0M2EsIFx1MDQzNVx1MDQ0MVx1MDQzYlx1MDQzOCBcdTA0MzJcdTA0NGIgXHUwNDNkXHUwNDM4XHUwNDQ3XHUwNDM1XHUwNDMzXHUwNDNlIFx1MDQzZFx1MDQzNSBcdTA0M2ZcdTA0M2VcdTA0M2JcdTA0NDNcdTA0NDdcdTA0MzBcdTA0MzVcdTA0NDJcdTA0MzU/In0sIHsidHlwZSI6ICJyYW5nZSIsICJyYW5nZSI6IFs1LCAxZSsxOF0sICJjb3JyZWN0IjogZmFsc2UsICJmZWVkYmFjayI6ICJcdTA0MTRcdTA0MzBcdTA0MzZcdTA0MzUgXHUwNDMyIFx1MDQxMFx1MDQ0MFx1MDQzM1x1MDQzNVx1MDQzZFx1MDQ0Mlx1MDQzOFx1MDQzZFx1MDQzNSBcdTA0M2RcdTA0MzUgXHUwNDNmXHUwNDNlXHUwNDNiXHUwNDQzXHUwNDQ3XHUwNDM4XHUwNDQyXHUwNDQxXHUwNDRmIFx1MDQ0MVx1MDQ0Mlx1MDQzZVx1MDQzYlx1MDQ0Y1x1MDQzYVx1MDQzZSBcdTA0MzdcdTA0MzBcdTA0NDBcdTA0MzBcdTA0MzFcdTA0M2VcdTA0NDJcdTA0MzBcdTA0NDJcdTA0NGMgXHUwNDNkXHUwNDMwIFx1MDQzNFx1MDQzNVx1MDQzZlx1MDQzZVx1MDQzN1x1MDQzOFx1MDQ0Mlx1MDQzNSJ9LCB7InR5cGUiOiAiZGVmYXVsdCIsICJmZWVkYmFjayI6ICJcdTA0MWRcdTA0MzVcdTA0NDIsIFx1MDQzNFx1MDQzNVx1MDQzZFx1MDQ0Y1x1MDQzM1x1MDQzOCBcdTA0M2JcdTA0NGVcdTA0MzFcdTA0NGZcdTA0NDIgXHUwNDQyXHUwNDNlXHUwNDQ3XHUwNDNkXHUwNDRiXHUwNDM5IFx1MDQ0MVx1MDQ0N1x1MDQ1MVx1MDQ0MiJ9XX1d</span>
