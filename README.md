# Ads Optmization Reinforcement

Reinforcement learning is part of the machine learning that train reward mechanisms
and punishment. The main objective is to take the best action or path possible to get the most
reward and minimal punishment through choice in a specific situation.

In this study, I reviewed the article "Reinforcement Learning: The Concept Behind UCB Explained With Code" [1]
and apply the concepts to an ad optimization database I found on Kaggle.

The main subject for using the Upper Confidence Bound (UCB) which focuses on two concepts: exploration and
exploitation. 

- exploration: gives equal chances of choice for all actions, regardless of their
average reward. Even stocks with the lowest average rewards chosen will be chosen.

- exploitation: is a technique that will always be looking to repeat the action that has already
presented the best result, being the one with the greatest reward, remembering
that we don't know which action gives the highest average reward of all possible.

## Problem

Find the Click Through Rate (CTR) of certain Ad.

## Tool used:
- Pandas
- Matplotlib
- Math
- Python

## Upper Confidence Bound (UCB)

UCB is a deterministic algorithm that means there is no uncertainty or probability factor, that is, the success rate distribution was calculated based on the probability distribution.

For a better understanding of the UCB it is necessary to understand the MultiArmed Bandit Problem.


## Implementation

The first thing, we need import our library that we will use our code.


import pandas as pd
import math
import matplotlib.pyplot as plt



After this, we need read our dataset with Pandas. We can do it with this command:


path = "/mydrive/data/"
df = pd.read_csv(path + "Ads_Optimization.csv")


### Data Exploration

Now, we can explore our dataset and see what we have there. I used tree things of pandas: shape, info and describe.


df.shape
(10000, 10)


With df.shape give our the shape our dataset, 1000 lines and 10 columns.


df.info()


<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 10 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   Ad 1    10000 non-null  int64
 1   Ad 2    10000 non-null  int64
 2   Ad 3    10000 non-null  int64
 3   Ad 4    10000 non-null  int64
 4   Ad 5    10000 non-null  int64
 5   Ad 6    10000 non-null  int64
 6   Ad 7    10000 non-null  int64
 7   Ad 8    10000 non-null  int64
 8   Ad 9    10000 non-null  int64
 9   Ad 10   10000 non-null  int64
dtypes: int64(10)
memory usage: 781.4 KB
None


With df.info give our information about our dataset.


df.describe()


               Ad 1          Ad 2  ...          Ad 9        Ad 10
count  10000.000000  10000.000000  ...  10000.000000  10000.00000
mean       0.170300      0.129500  ...      0.095200      0.04890
std        0.375915      0.335769  ...      0.293506      0.21567
min        0.000000      0.000000  ...      0.000000      0.00000
25%        0.000000      0.000000  ...      0.000000      0.00000
50%        0.000000      0.000000  ...      0.000000      0.00000
75%        0.000000      0.000000  ...      0.000000      0.00000
max        1.000000      1.000000  ...      1.000000      1.00000

[8 rows x 10 columns]


And with df.describe() we have a little statistic just with this simple command. 

### Upper Confidence Bound code


number_ads = 10 # numbers of ads (number of columns in this case)
observations = 1000  # number of observations ( number of lines in this case)
numbers_selections = [0] * number_ads  # initialing our variable of number of selections of each ads
sum_rewards = [0] * number_ads    # initializing our variable of sums of rewards of each ads
total_reward = 0
selected_ads = []

for n in range(0, observations):
  ads = 0
  max_upper_bound = 0
  for i in range(0, number_ads):

    if numbers_selections[i] > 0:
      average = sum_rewards[i]/numbers_selections[i]      # first formule
      delta = math.sqrt(3/2 * math.log(n + 1) / numbers_selections[i])    # second formule
      upper_bound = average + delta    # third formule
    else:
      upper_bound = 1e400
    if upper_bound > max_upper_bound:
      max_upper_bound = upper_bound
      ads = i

  selected_ads.append(ads)
  numbers_selections[ads]  = numbers_selections[ads] + 1
  reward = df.values[n, ads]
  sum_rewards[ads] = sum_rewards[ads] + reward
  total_reward = total_reward + reward


### Visualizing results

### References:

[1] https://analyticsindiamag.com/ad-click-through-rate-ctr-prediction-using-reinforcement-learning/
