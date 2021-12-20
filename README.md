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

TechTalks [2] explain the name of the MultiArmed Bandit Problem comes from an imaginary scenario where a player is in a row of slot machines. The player knows that machines have different win rates, but doesn't know which one offers the greatest reward.

If he limits himself to one machine, he may lose the chance to select the machine with the highest win rate. Therefore, the player must find an efficient way to discover the machine with the highest reward without spending too much of their tokens.

Ad optimization is a typical example of a multi-armed bandit problem. In this case, the reinforcement learning agent must find a way to discover the ad with the highest CTR without wasting too much valuable ad impressions on inefficient ads.

### UCB Math

Next is the algorithm inside the UCB that updates the confidence limits of each machine after each round. Let's see the explanation of  AMAL NAIR [3].

Step 1: Two values are considered for each round of exploration of a machine
The number of times each machine has been selected till round n
The sum of rewards collected by each machine till round n

Step 2: At each round, we compute the average reward and the confidence interval  of the machine i up to n rounds as follows:

Average reward :

![Average reward](https://github.com/Riquinho93/Ads-Optmization-Reinforcement/blob/main/assets/first.png)

Confidence Interval :

![Confidence Interval](https://github.com/Riquinho93/Ads-Optmization-Reinforcement/blob/main/assets/second.png)

Step 3: The machine with the maximum UCB is selected.

UCB:

![UCB](https://github.com/Riquinho93/Ads-Optmization-Reinforcement/blob/main/assets/third.png)

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
> (10000, 10)


With df.shape give our the shape our dataset, 1000 lines and 10 columns.


        df.info()
>        <class 'pandas.core.frame.DataFrame'>
>        RangeIndex: 10000 entries, 0 to 9999
>        Data columns (total 10 columns):
>        #   Column  Non-Null Count  Dtype
>         ---  ------  --------------  -----
>        0   Ad 1    10000 non-null  int64
>        1   Ad 2    10000 non-null  int64
>        2   Ad 3    10000 non-null  int64
>        3   Ad 4    10000 non-null  int64
>        4   Ad 5    10000 non-null  int64
>        5   Ad 6    10000 non-null  int64
>        6   Ad 7    10000 non-null  int64
>        7   Ad 8    10000 non-null  int64
>        8   Ad 9    10000 non-null  int64
>        9   Ad 10   10000 non-null  int64
>        dtypes: int64(10)
>        memory usage: 781.4 KB
>        None


With df.info give our information about our dataset.


          df.describe()
>                      Ad 1          Ad 2  ...          Ad 9        Ad 10
>         count  10000.000000  10000.000000  ...  10000.000000  10000.00000
>         mean       0.170300      0.129500  ...      0.095200      0.04890
>         std        0.375915      0.335769  ...      0.293506      0.21567
>         min        0.000000      0.000000  ...      0.000000      0.00000
>         25%        0.000000      0.000000  ...      0.000000      0.00000
>         50%        0.000000      0.000000  ...      0.000000      0.00000
>         75%        0.000000      0.000000  ...      0.000000      0.00000
>         max        1.000000      1.000000  ...      1.000000      1.00000

>         [8 rows x 10 columns]


And with df.describe() we have a little statistic just with this simple command. 

### Upper Confidence Bound code

Now let's start our algorithm, let's iterate over each machine on each observation starting with B1 (indexed 0) and with a maximum upper bound value of zero.

At each round, we will check if a machine was selected before or not. If so, the algorithm proceeds to calculate the machine's average rewards, delta, and superior confidence. Otherwise, ie if the machine is being selected for the first time, it sets a default upper limit value of 1e400.

After each round, the machine with the highest upper limit value is selected, the number of selections along with the actual reward and the sum of rewards for the selected machine are updated.

After all rounds are completed, we will have a machine with a maximum value of the upper limit. See the code below:


       
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

Rewards by ads:

           print("Rewards by ads: ", sum_rewards)
> Rewards by ads:  [17, 8, 1, 5, 79, 1, 10, 27, 4, 3]

Rewards by UCB:

           print("Rewards by UCB: ", total_reward)
> Rewards by UCB:  155

Ads selected:

       print("Ads selected: ", selected_ads)
> Ads selected:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 0, 8, 1, 2, 3, 4, 5, 6, 6, 7, 7, 7, 9, 7, 0, 6, 8, 1, 2, 3, 4, 4, 4, 4, 4, 5, 9, 7, 0, 0, 6, 8, 4, 0, 7, 1, 1, 2, 3, 5, 9, 4, 1, 6, 8, 0, 7, 2, 3, 5, 9, 4, 1, 6, 8, 0, 7, 7, 7, 7, 7, 7, 4, 2, 3, 5, 9, 1, 6, 8, 0, 0, 7, 0, 4, 1, 6, 8, 2, 3, 5, 9, 7, 0, 4, 4, 4, 7, 7, 7, 7, 1, 6, 8, 2, 3, 5, 9, 0, 7, 7, 4, 7, 7, 7, 1, 6, 8, 0, 4, 4, 2, 3, 5, 9, 4, 7, 0, 7, 1, 1, 1, 6, 6, 6, 6, 6, 8, 4, 2, 3, 5, 9, 7, 7, 7, 0, 6, 4, 1, 1, 1, 8, 7, 2, 3, 5, 9, 4, 0, 1, 1, 1, 6, 7, 8, 4, 1, 2, 3, 5, 9, 0, 6, 7, 8, 4, 7, 1, 0, 6, 2, 3, 5, 9, 7, 7, 7, 7, 4, 4, 4, 4, 4, 7, 8, 1, 0, 6, 6, 6, 7, 4, 2, 3, 5, 9, 1, 6, 8, 7, 0, 4, 4, 4, 4, 4, 2, 3, 5, 9, 1, 6, 7, 4, 8, 0, 7, 4, 4, 4, 1, 6, 2, 3, 5, 9, 0, 0, 0, 7, 8, 4, 0, 0, 0, 1, 6, 7, 4, 2, 3, 5, 9, 0, 8, 7, 1, 6, 4, 0, 7, 7, 7, 7, 7, 2, 3, 5, 9, 8, 4, 1, 6, 6, 6, 7, 7, 7, 0, 6, 7, 4, 1, 1, 1, 2, 2, 2, 3, 5, 9, 2, 8, 7, 0, 1, 6, 4, 7, 2, 8, 3, 5, 9, 9, 9, 9, 4, 0, 1, 6, 7, 4, 7, 0, 1, 6, 2, 8, 9, 3, 5, 4, 7, 0, 0, 0, 1, 6, 2, 8, 9, 4, 0, 3, 5, 7, 1, 6, 7, 4, 0, 2, 8, 9, 3, 5, 7, 4, 4, 4, 4, 4, 1, 6, 4, 0, 7, 2, 8, 9, 4, 3, 5, 1, 6, 6, 6, 0, 6, 7, 4, 4, 4, 2, 8, 8, 8, 8, 9, 4, 1, 7, 0, 6, 3, 5, 7, 7, 7, 4, 8, 1, 2, 9, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 5, 4, 4, 4, 7, 4, 4, 4, 0, 1, 1, 1, 8, 8, 8, 8, 1, 6, 4, 2, 9, 7, 3, 3, 3, 3, 5, 0, 4, 4, 4, 8, 1, 6, 4, 7, 2, 3, 9, 0, 5, 4, 7, 1, 6, 8, 0, 2, 3, 9, 4, 7, 7, 7, 7, 1, 6, 8, 5, 0, 4, 7, 2, 3, 9, 1, 6, 4, 0, 8, 7, 5, 4, 1, 6, 6, 6, 6, 2, 3, 3, 3, 3, 9, 7, 0, 8, 4, 4, 4, 4, 4, 4, 3, 5, 7, 6, 1, 1, 1, 1, 0, 4, 2, 9, 8, 8, 8, 8, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 1, 6, 4, 4, 4, 0, 5, 4, 4, 4, 8, 7, 7, 7, 2, 9, 4, 7, 0, 1, 6, 3, 4, 5, 5, 5, 5, 7, 8, 2, 5, 9, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 1, 6, 4, 7, 7, 7, 3, 7, 8, 4, 0, 1, 6, 2, 5, 9, 9, 9, 9, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 9, 4, 8, 0, 0, 0, 0, 0, 0, 0, 7, 1, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 2, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 3, 9, 4, 0, 8, 6, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7, 2, 5, 7, 7, 7, 0, 0, 0, 0, 4, 7, 6, 3, 9, 8, 1, 1, 1, 1, 4, 7, 0, 4, 4, 4, 2, 5, 1, 6, 4, 4, 4, 4, 4, 4, 4, 4, 8, 7, 4, 3, 9, 0, 4, 7, 1, 6, 6, 6, 6, 2, 5, 4, 4, 4, 4, 4, 4, 4, 4, 8, 4, 0, 3, 3, 3, 3, 9, 7, 4, 6, 3, 1, 7, 4, 0, 2, 5, 8, 9, 4, 4, 4, 4, 4, 4, 6, 7, 1, 3, 4, 4, 4, 0, 4, 8, 7, 2, 5, 6, 6, 6, 6, 4, 9, 1, 0, 4, 3, 7, 6, 4, 8, 0, 2, 5, 7, 1, 9, 9, 9, 9, 4, 6, 3, 3, 3, 3, 9, 4, 4, 4, 7, 4, 0, 0, 0, 0, 3, 8, 1, 2, 5, 4, 7, 7, 7, 6, 7, 0, 9, 4, 3, 8, 1, 7, 7, 7, 7, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 4, 9, 7, 0, 1, 4, 3, 8, 6, 7, 4, 0, 2, 5, 9, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 6, 3, 8, 4, 0, 4, 7, 2, 5, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 9, 6, 4, 4, 4, 4, 4, 4, 0, 3, 3, 3, 3, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 3, 4, 1, 6, 2, 5, 0, 9, 4, 4, 4, 7, 4, 8, 4, 3, 7, 0, 1, 6, 4, 2, 5, 9]

### Visualize in the histogram

          plt.hist(selected_ads)
          plt.title("Histogram Ads selected")
          plt.xlabel("Ads")
          plt.ylabel("Number of times")
          plt.show()

![Ads](https://github.com/Riquinho93/Ads-Optmization-Reinforcement/blob/main/assets/ads.png)

### References:

[1] https://analyticsindiamag.com/ad-click-through-rate-ctr-prediction-using-reinforcement-learning/

[2] TECHTALKS. How reinforcement learning chooses the ads you see. Available in: <https://bdtechtalks.com/2021/02/22/reinforcement-learning-ad-optimization/>.

[3] ANALYTICSINDIAMAG. Reinforcement Learning : The Concept  Behind UCB Explained With Code. Available in:<https://analyticsindiamag.com/reinforcement-learning-the-concept-behind-ucb-explained-with-code/>
