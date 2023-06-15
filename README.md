# Who Wins the Match Based on the Game State at 15 minutes?

This is Project 5 for DSC80 at UCSD.

---
Our exploratory data analysis on this dataset can be found [here](https://liang-justin.github.io/professional-lol-leads/). Some additional information has been taken such as descriptions.


## Framing the Problem

League of Legends (LOL) is a popular MOBA that is played by many people around the world. Like other sports like basketball, football, soccer, etc. it can be enjoyed casually and even professionally in the form of League of Legends Esports. Like the many professional leagues for sports, League of Legends Esports is a collection of the best players and teams in the world which compete to lift the Summoner’s Cup.

With its large increase in popularity, fans, both serious and casual, are able to tune in to these matches and see their favorite teams and players parttake in such events. In a similar fashion to other sports leagues, many of the important events and statistics in each match are documented for those to see and understand. Information such as gold earned, xp gained, CS, etc. can help us viewers understand the current climate of the game and provide insight to who might be victorious at the end. This information is not only valuable to the viewer, but can also be valueable to the players and teams on ways to improve in the future.

In regular casual matches, players at given the option to forfeit the game at the 15 minute mark in game if they determine it is reasonable to do so. However, professional players aren't as fortunate. Using these metrics of a professional match, we are hoping to use the data collected in the 2022 Professional League of Legends Season (dataset organized by OraclesElixir) to train a model and use it to predict unseen match data in hopes of determining who the winner is.

We will be exploring the impact of game statistics at the 15 minute mark in professional League of Legends matches to determine which team will win the match at the end of the game. We will be using a Binary Classification to determine this as we are predicting either a win or a loss from the game statistics.

- **The Response Variable**: Predict which team wins based on Game Statistics at the 15 minute mark.

The metric that I will be using to evaluate this model is accuracy. Accuracy is a good evalution metric for this model as in a single Professional League of Legends match, there is always going to be a winning team and losing team. This makes the other metrics (precision, F1-Score, etc.) not a very good metric to evaluate this model since the data is quite balanced - that with every team that wins, there is always a team that loses per `gameid`.

## Baseline Model

The Baseline Model is a Binary Classification based on the game statistics: Gold differential at 15 minutes, the team that killed the first Dragon, and the Team that killed the first Rift Herald. We will be using a `LogisticRegression model` from the `sklearn` module.

The data classification for these columns is as follows:
- Gold Differential at 15 minutes (`'golddiffat15'`): Quantitative Continuous
- Team that got First Dragon Kill (`'firstdragon'`): Qualitative Nominal
- Team that got First Herald Kill (`'firstherald'`): Qualitative Nominal

The encodings that were done to the data were performed by standardizing the `'golddiffat15'` column since we want to minimize the number of outliers (e.g. large leads that are very positive/very negative for the respective team) and make the data comparable to other data in the future by using `sklearn` module `StandardScaler` package. Additional transformations on the remaining two features were unneeded as they were already in a "one-hot" like format (boolean).

I think that the baseline model prediction of the winner of matches performed alright. With an accuracy of ~0.74, I don't think it truly is able to give a very good prediction of who killed the first baron given the limited number of features as they don't truly capture the game state at the 15 minute mark. This is because these three metrics only give a small glimpse of what the game state is like. Even though one team is able to get the first herald or the first dragon, it doesn't suggest that the team is currently leading or in a better state than the other. In other words, there could be a multitude of variables that contributed to which team killed the first dragon/herald.

I believe that the addition of other metrics at the 15 minute mark could help the model make a better prediction on who will win the game.

## Final Model

When compared to the baseline model, the additional information that was added to the model were the rest of the metrics that are commonly measured at the 15 minute mark in a common League of Legends Match, namely `'goldat15'`, `'xpat15'`, `'csat15'`,`'xpdiffat15'`, `'csdiffat15'`, `'killsat15'`, `'assistsat15'`, `'deathsat15'`. As stated earlier, these metrics help paint a picture to a viewer on the state of the game at a common mark in the game where champions are beginning to reach their powerspikes. Unlike the baseline model, which only used the `'golddiffat15'`, `'firstherald'`, `'firstdragon'`, and `'heralds'` data to predict the outcome of the match, these features might not properly describe the condition of the game as these statistics can sometimes only be taken at face value.

More specifically, the columns of `'xpdiffat15'` and `'csdiffat15'` are also important metrics in a game of League of Legends. As the the gold earned by each champion influencing what they able to purchase from the store after recalling, the differences in XP and CS (creep score) also help to paint a picture of how the team is doing compared to the other in terms of prowess on the rift -lane control, vision control, etc. - and control of the direction of the game - pace, pushing, etc..

Columns of `'goldat15'`, `'xpat15'`, `'csat15'`, `'killsat15'`, `'assistsat15'`, and `'deathsat15'` also help to describe the condition of the game for each team at the 15 minute mark. Unlike the differentials above, this data is helpful in the sense of gauging the condition of the game. Is it bloody? Is it a stalemate with no kills for each side? Will this game take long, or will it be a swift sweep? Moreover, the differential metrics might misrepresent the condition of the game: let's say a jungler has been camping on of the lanes rather than farming their jungle, this means that they might have equal gold but be down in a lot of XP, which might mean that the data reflects that they might just be behind in general. These metrics are like the main course, with the differentials being the sauce that enhances the dish. 

A `GridSearchCV` was performed using a set of hyperparameters for this model. Since there are multiple metrics that are not compatible - certain `'penalty'` hyperparameters like `'l1'` not compatible with certain `'solver'` hyperparameters like `'saga'`, we chose to just use compatible hyperparameters in the GridSearch function. The training data was then fit to the GridSearchCV model and the most optimized hyperparameters were determined to be `'{'C': 0.1, 'max_iter': 10000, 'penalty': 'l2', 'solver': 'sag'}'` 

A `LogisticRegresion` model was again used to determine the winner of the match based on the metrics outlined above. The Final Model did seem to perform better when compared to the Baseline model. This might have been attributed to the extra features that were used to train the model since they help to describe the game state a little better at the 15 minute mark. I think the fault in the model might be due to the nature of the games after the 15 minute mark since in professional matches they can go either way even though a team might have lead/deficit due to the the many powerspikes that accomany each champ that is in the meta at the time. This will be further discussed below.

## Fairness Analysis

The side that each team plays on might have some sway in who wins the match at the end. Like StarCraft with one side being 'usually' stronger than the other, we want to explore how the slight differences between the side selection - banning champ sequence, choosing sequence, etc. have an effect on our model's accuracy.

Two Groups:
- Blue Side
- Red Side

Our Permutation test will be as follows:

- **Null Hypothesis:** The accuracy of our model predicted that both Blue and Red Side are roughly the same. The differences in the accuracy is due to random error.

- **Alternative Hypothesis**: Our model isn't fair. It is more accurate for Teams on Blue Side rather than Teams on Red Side.

Our significance level will be `0.05`. And our test statistic will be the signed differenced in accuracy between the predicted game outcome on Blue-side and teams on Red-side.

Since the number of Blue-Sided Teams and Red-Sided teams are roughly the same - one team is always Blue Side and one team is always Red Side - we will be using Accuracy for the evaluation metric.

Using the determined signed differences between the accuracy between the match winner predictions, we wanted to determine how many values fell above the observed accuracy difference. Our determined p-value was 0.64 which is much greater than the significance level of 0.05. This suggests that the side chosen doesn't truly effect the model's performance, which is to be expected. As a larger esport in the scene, League of Legends seemingly is trying to make sure that some of the variables that occur in the game allow for the expression of skill and stragtegy rather than just straight luck. Especially for the side determination, if there was a large discrepancy between sides, our model wouldn't probably not be able to predict the winner of each match due to the addition of an extraneous variable in the path to victory for teams. 