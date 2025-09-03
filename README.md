# Horse-Racing-Project

In this project I intend to backtest a betting strategy. I have uploaded then cleaned a kaggle dataset, it is too big to put in GitHub. But for my final model it is under models -> lightGGMB.ipynb. One problem I faced was that the market odds were clearly made after the race so my algorithm's PnL is very high.

Listed below are the steps I took in completing this Project:
1. Collected data from kaggle, cleaned features including missing data.
2. Feature engineered new stats. E.g. Days since last race, historical performance on race track, jockey etc. Overall there were over 200 features originally 50. Also got rid of redundant features, one hot encoded or numerically encoded data types.

3. Initially created a Linear Regressionas as my baseline.
4. Created LightGBM due to large amount of data, then created a catBoost which allowed for better categroical and unique data types. E.g. horseName which has a lot of different values. Used Log Loss, Brier Score, RMSE as metrics for this. Implemented train/test with temporal split. (1997-2015) Training set, (2015-2018) Training Set.
5. Decided to go with only LightGBM, calibrated this model using isotonic regression and platt sclaing to make probabilities more accurate, tested this on the year 2019.
6. Backtested the strategy on 2020 data, to get a positive PnL. Unfortunately found my algorithm significantly had a profit due, initially thought data leakage then found out that proability errors were in data as the market win percentage were taken after the race. Unable to find similar dataset.

   Conclusions:

   I created a good model in my opinion, especially for the large amount of data (over 3 million rows, 200 features) it is  best to go with a lightGBM or CatBoost algorithm (if using catBoost would recommend using external GPU e.g. google Colab). I was quite produ of putting the data into time order, as I thought this was a good challenge espeically applicable to finance data where data leakage can occur if data is not handled properly.

   However, if I were to do it again. I would maybe challenge myself to find more realistic data, perhaps by webscraping. Even though this would be much harder and make not a great result I think it would be a great challenge to see if I could get my model to work in the real world. By using real-world data I would have to focus less on the model and more on how to make my data better, but it would let me apply it to real life. Despite this my algorithm worked really well. I think one 
