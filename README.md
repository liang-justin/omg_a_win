# Who Wins the Match Based on the Game State at 15 minutes?

This is Project 5 for DSC80 at UCSD.

---
Our exploratory data analysis on this dataset can be found [here](https://liang-justin.github.io/professional-lol-leads/). Some additional information has been taken such as descriptions.


## Framing the Problem

League of Legends (LOL) is a popular MOBA that is played by many people around the world. Like other sports like basketball, football, soccer, etc. it can be enjoyed casually and even professionally in the form of League of Legends Esports. Like the many professional leagues for sports, League of Legends Esports is a collection of the best players and teams in the world which compete to lift the Summoner’s Cup.

With its large increase in popularity, fans, both serious and casual, are able to tune in to these matches and see their favorite teams and players parttake in such events. In a similar fashion to other sports leagues, many of the important events and statistics in each match are documented for those to see and understand. Information such as gold earned, xp gained, CS, etc. can help us viewers understand the current climate of the game and provide insight to who might be victorious at the end. This information is not only valuable to the viewer, but can also be valueable to the players and teams on ways to improve in the future.

In regular casual matches, players at given the option to forfeit the game at the 15 minute mark in game if they determine it is reasonable to do so. However, professional players aren't as fortunate. Using these metrics of a professional match, we are hoping to use the data collected in the 2022 Professional League of Legends Season to train a model and use it to predict unseen match data in hopes of determining who the winner is.


We will be exploring the impact of game statistics at the 15 minute mark in professional League of Legends matches to determine which team will win the match at the end of the game.

- **The Response Variable**: Predict which team wins based on Game Statistics at the 15 minute mark.

The metric that I will be using to evaluate this model is accuracy. Accuracy is a good evalution metric for this model as in a single Professional League of Legends match, there is always going to be a winning team and losing team. This makes the other metrics (precision, F1-Score, etc.) not a very good metric to evaluate this model since the data is quite balanced - that with every team that wins, there is always a team that loses per `gameid`.