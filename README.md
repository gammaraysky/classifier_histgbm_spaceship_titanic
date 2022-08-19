# spaceship_titanic_classifier
EDA and classifier modelling for Kaggle 'Spaceship Titanic' Playground Competition

### Brief
During the journey, the spaceship collided with a space-time anomaly resulting in some passengers being 'transported' and have disappeared. The task is to create a predictive model to predict which passengers may have 'transported' in the unlabelled test data.

### Summary
- EDA on dataset
- **Missing Data**
  - Imputing data based on EDA analysis
    - **Continuous Data:**
      - Those in CryoSleep did not spend any money on RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
      - Children aged 12 and under did not spend any money on RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
    - **Categorical Data:**
      - PassengerId are in the format 0123_01, 0123_02, for passengers travelling as groups.
        - This allows us to impute some missing values for HomePlanet, Destination, Cabin, VIP
      - for missing values in CryoSleep, take it under the assumption that passengers requesting for CryoSleep would have to be pre-registered, thus missing values filled with False.
  - Imputing the rest of the numerical data with KNNImputer
  - For the remaining missing categoricals, leave it as-is, as OneHotEncoder will by default treat it as it's own category later on during preprocessing (from sklearn 0.24 onwards). 
- **Feature Engineering**
  - groupsize engineered from value counts of PassengerId prefixes
  - Cabin is in the format of Deck/Number/Side (e.g. A/1/P or B/300/S) where side is port or starboard.
    - cabin_deck, cabin_number, cabin_side are parsed out with regex.
    - cabin_number ranges from 1-1800+, so it's binned into groups of 100
  - Age is binned into groups of 3 years
  - Columns RoomService, FoodCourt, ShoppingMall, Spa, VRDeck were summed into sumspend, and for all of these, log-transformed values were also created. Also, the value ranges are too wide, with most people spending $0 but some spent as much as $20,000 at individual spots. Thus we used RobustScaler later on.
- **Model Selection**
  - using sklearn pipelines, we use RobustScaler on numerical values and OneHotEncoder and OrdinalEncoder(only for HGBM)
  - attempted GaussianNB, LogisticRegression, SVC, AdaBoostClassifier, RandomForestClassifier, HistGradientBoostingClassifier (with slight tuning)
  - attempted PCA to hopefully denoise/compress/reduce multicollinearity, and tried with the above classifiers.
  - selected HistGradientBoostingClassifier (using native categorical support, so passed in ordinally encoded columns, but it will be treated as nominal) as best performing model, tuned further, refit with entire train dataset without holdout, and predicted on test.csv for Kaggle submission.
  - achieved 0.80576 accuracy score on Kaggle, leaderboard position 256/2350.
  
