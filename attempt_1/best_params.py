lgbm_params = {
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

xgb_params ={
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
# 'learning_rate': 0.040289198629689146,
#     'max_depth': 10,
#     'subsample': 0.8558982535197237,
#     'colsample_bytree': 0.6428358438534396,
#     'reg_lambda': 2.991103556078565e-06,
#     'min_child_weight': 6,
#     'gamma': 0.00016968006054236202
}


cb_params = {'objective': 'RMSE',
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
    "PlayoutsPerSecond", "MovesPerSecond", "p1_selection",
    "p1_exploration", "p1_playout", "p1_bounds", "p2_selection", "p2_exploration",
    "p2_playout", "p2_bounds", #'char_count_english'
]