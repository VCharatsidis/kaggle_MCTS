import catboost
import lightgbm
import xgboost as xgb

import polars as pl
import pandas as pd
import numpy as np


class Preprocessor():
    def __init__(self):
        global cat_features
        global best_features

        self.lgbm_params = {
            'boosting_type': 'gbdt',
            'learning_rate': 0.08323424641333231,
            'n_estimators': 300,
            'max_depth': 8,
            'subsample': 0.7178970226547821,
            'colsample_bytree': 0.8410522505616279,
            'reg_lambda': 1.1157041999759082e-06,
            'min_child_samples': 27,
            'num_leaves': 2382,
            'objective': 'mse',
            'subsample_freq': 1,
            'verbose': -1
        }

        self.xgb_params = {
            'enable_categorical': True,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'learning_rate': 0.0570718601460729,
            'max_depth': 10,
            'subsample': 0.8977779092288535,
            'colsample_bytree': 0.71785867753211,
            'reg_lambda': 1.8125268205502734e-08,
            'min_child_weight': 9,
            'gamma': 7.977924974233929e-05,
            'n_estimators': 300,

        }

        self.cb_params = {'objective': 'RMSE',
                          'eval_metric': 'RMSE',
                          'learning_rate': 0.1575265989034749,
                          'depth': 10,
                          'subsample': 0.6031520424053429,
                          'min_data_in_leaf': 48,
                          'verbose': 0,
                          'iterations': 200
                          }

        best_features = [
            "Asymmetric", "MorrisTiling", "ConcentricTiling", "MancalaBoard", "MancalaStores",
            "AlquerqueBoard", "Region", "NumConvexCorners", "NumDice", "NumStartComponentsHand",
            "SwapPlayersDecision", "AddDecisionFrequency", "RemoveDecision", "RemoveDecisionFrequency",
            "StepDecision", "SlideDecisionFrequency", "HopDecisionEnemyToEmpty",
            "FromToDecisionBetweenContainersFrequency", "FromToDecisionFriendFrequency",
            "RollFrequency", "SowFrequency", "SowWithEffect", "SowCaptureFrequency",
            "SetNextPlayerFrequency", "StepEffect", "Priority", "HopCaptureFrequency",
            "HopCaptureMoreThanOneFrequency", "CustodialCaptureFrequency", "Line",
            "OrthogonalDirection", "Phase", "PieceCount", "SpaceEnd", "LineEndFrequency",
            "LineWinFrequency", "ConnectionEndFrequency", "ConnectionWin", "ConnectionWinFrequency",
            "CheckmateFrequency", "CheckmateWin", "CheckmateWinFrequency", "EliminatePiecesEndFrequency",
            "EliminatePiecesWin", "EliminatePiecesWinFrequency", "ReachWinFrequency", "NoMovesDraw",
            "DurationActions", "DurationTurnsStdDev", "DurationTurnsNotTimeouts", "DecisionMoves",
            "GameTreeComplexity", "AdvantageP1", "Completion", "OutcomeUniformity", "BoardSitesOccupiedAverage",
            "BoardSitesOccupiedMaximum", "BoardSitesOccupiedVariance", "BoardSitesOccupiedChangeAverage",
            "BoardSitesOccupiedChangeSign", "BoardSitesOccupiedMaxIncrease", "BoardSitesOccupiedMaxDecrease",
            "BranchingFactorAverage", "BranchingFactorMedian", "BranchingFactorMaximum",
            "BranchingFactorChangeAverage", "BranchingFactorChangeMaxIncrease", "BranchingFactorChangeMaxDecrease",
            "DecisionFactorMedian", "DecisionFactorVariance", "DecisionFactorChangeSign",
            "DecisionFactorMaxIncrease", "DecisionFactorMaxDecrease", "MoveDistanceChangeSign",
            "MoveDistanceChangeLineBestFit", "PieceNumberMedian", "PieceNumberVariance",
            "PieceNumberChangeAverage", "PieceNumberChangeSign", "LesserThanOrEqual", "GreaterThanOrEqual",
            "Disjunction", "CheckersComponent", "State", "PieceState", "SiteState", "ForEachPiece",
            "PlayoutsPerSecond", "MovesPerSecond", "char_count_english", "p1_selection",
            "p1_exploration", "p1_playout", "p1_bounds", "p2_selection", "p2_exploration",
            "p2_playout", "p2_bounds"
        ]

        cat_features = ['p1_selection', 'p1_playout', 'p1_bounds', 'p2_selection', 'p2_playout', 'p2_bounds']

        self.train_models()

    def preprocess(self, df_polars):
        """Convert the polars dataframe to pandas; extract target and groups if it is the training dataframe

        The function should be applied to training and test datasets.

        Parameters
        df_polars: polars DataFrame (train or test)

        Return values:
        df: pandas DataFrame with all features of shape (n_samples, n_features)
        target: target array of shape (n_samples, ) or None
        groups: grouping array for GroupKFold of shape (n_samples, ) or None
        """

        df_polars = df_polars.with_columns(pl.col("EnglishRules").str.len_chars().alias("char_count_english"))
        df_polars = df_polars.with_columns(pl.col("LudRules").str.len_chars().alias("char_count_lud"))

        # Add eight features extracted from player names,
        # Drop GameRulesetName, freetext and target columns
        df = df_polars.with_columns(
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 1).alias('p1_selection'),
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 2).alias('p1_exploration').cast(pl.Float32),
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 3).alias('p1_playout'),
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 4).alias('p1_bounds'),
            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 1).alias('p2_selection'),
            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 2).alias('p2_exploration').cast(pl.Float32),
            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 3).alias('p2_playout'),
            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 4).alias('p2_bounds')
        ).drop(
            ['GameRulesetName', 'EnglishRules', 'LudRules']
        ).to_pandas()

        for col in cat_features:
            df[col] = df[col].astype('category')

        return df

    def train_models(self):
        train = pl.read_csv('um-game-playing-strength-of-mcts-variants/train.csv')
        num_rows = len(train)
        cutoff = int(num_rows * 0.8)
        df_first_80_percent = train.slice(0, cutoff)
        train_pd = self.preprocess(df_first_80_percent)

        global lgb_model, cb_model, xgb_model

        X_train = train_pd[best_features]
        y_train = train_pd['utility_agent1']

        lgb_model = lightgbm.LGBMRegressor(**self.lgbm_params)
        lgb_model.fit(X_train, y_train)

        cb_model = catboost.CatBoostRegressor(**self.cb_params, cat_features=cat_features)
        cb_model.fit(X_train, y_train)

        xgb_model = xgb.XGBRegressor(**self.xgb_params)
        xgb_model.fit(X_train, y_train)

    def infer(self, test, best_features, cat_features, lgb_model, cb_model, xgb_model):
        test = test.with_columns(pl.col("EnglishRules").str.len_chars().alias("char_count_english"))
        test = test.with_columns(pl.col("LudRules").str.len_chars().alias("char_count_lud"))

        # Add eight features extracted from player names,
        # Drop GameRulesetName, freetext and target columns
        df = test.with_columns(
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 1).alias('p1_selection'),
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 2).alias('p1_exploration').cast(pl.Float32),
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 3).alias('p1_playout'),
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 4).alias('p1_bounds'),
            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 1).alias('p2_selection'),
            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 2).alias('p2_exploration').cast(pl.Float32),
            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 3).alias('p2_playout'),
            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 4).alias('p2_bounds')
        ).drop(
            ['GameRulesetName', 'EnglishRules', 'LudRules']
        ).to_pandas()

        for col in cat_features:
            df[col] = df[col].astype('category')

        X_test = df[best_features]

        lgb_y_pred = lgb_model.predict(X_test)
        cb_y_pred = cb_model.predict(X_test)
        xgb_y_pred = xgb_model.predict(X_test)

        y_pred = (lgb_y_pred + cb_y_pred + xgb_y_pred) / 3

        return y_pred

from sklearn.metrics import mean_squared_error
def predict(test):
    global counter
    global preprocessor

    preprocessor = Preprocessor()

    y_pred = preprocessor.infer(test, best_features, cat_features, lgb_model, cb_model, xgb_model)
    y_train = test['utility_agent1']
    loss = np.sqrt(mean_squared_error(y_train, y_pred))

    print(loss)


train = pl.read_csv('um-game-playing-strength-of-mcts-variants/train.csv')
num_rows = len(train)
cutoff = int(num_rows * 0.8)
df_first_80_percent = train.slice(cutoff, num_rows-cutoff)

predict(df_first_80_percent)
