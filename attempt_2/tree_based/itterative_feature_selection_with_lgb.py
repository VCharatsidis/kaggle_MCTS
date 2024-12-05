
import lightgbm
import polars as pl
import pandas as pd
import numpy as np
from colorama import Fore, Style
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from train_utils import preprocess, get_test_indexes_from_file

test = pl.read_csv('../um-game-playing-strength-of-mcts-variants/test.csv')
print('Shape before dropping columns:', test.shape)

train = pl.read_csv('../um-game-playing-strength-of-mcts-variants/train.csv')
print('Shape before dropping columns:', train.shape)

# custom_split(train, 6)
# input()

constant_columns = np.array(train.columns)[train.select(pl.all().n_unique() == 1).to_numpy().ravel()]
print(len(constant_columns), 'columns are constant. These will be dropped.')
drop_columns = list(constant_columns)
train = train.drop(drop_columns)
print('Shape after dropping columns:', train.shape)

train_pd, y, groups, cat_mapping = preprocess(train)

# All the game features (concepts) have a ComputationTypeId, which is either 'Compiler' or 'Simulation'
concepts = pd.read_csv('../../um-game-playing-strength-of-mcts-variants/concepts.csv', index_col='Id')
concepts[['TypeId', 'DataTypeId', 'ComputationTypeId', 'LeafNode', 'ShowOnWebsite']] = concepts[['TypeId', 'DataTypeId', 'ComputationTypeId', 'LeafNode', 'ShowOnWebsite']].astype(int)
concepts.replace({'ComputationTypeId': {1: 'Compiler', 2: 'Simulation'}}, inplace=True)
# print(concepts.ComputationTypeId.value_counts())


features = [f for f in train_pd.columns if f not in ['agent1', 'agent2', 'Id', 'num_wins_agent1', 'num_draws_agent1','num_losses_agent1', 'utility_agent1']]

lgbm_params = {'boosting_type': 'gbdt', 'learning_rate': 0.03, 'n_estimators': 200 ,'max_depth': 8,
               'objective': 'mse', 'verbose': -1}


# lgbm_params = {'objective': 'regression',
#         'metric': 'mse',
#         'boosting_type': 'gbdt',
#         'n_estimators': 300,
#         'verbose': -1,
# 'learning_rate': 0.0797353804621146, 'max_depth': 10, 'subsample': 0.8844130355984797, 'colsample_bytree': 0.7496776024413567, 'reg_lambda': 0.6889544202705858, 'min_child_samples': 28, 'num_leaves': 1863, 'subsample_freq': 3
#               }

test_ids = get_test_indexes_from_file('data_splits/test_set.txt')
test_data = train_pd[train_pd['Id'].isin(test_ids)]


def predict_sub_set(test_fold, train_fold, lgbm_params):
    X_test = test_fold[features]
    y_test = test_fold['utility_agent1']

    X_train = train_fold[features]
    y_train = train_fold['utility_agent1']

    model = lightgbm.LGBMRegressor(**lgbm_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return test_rmse, model, X_test, y_test


test_ids = get_test_indexes_from_file('data_splits/test_set.txt')
test_data = train_pd[train_pd['Id'].isin(test_ids)]
general_train_fold = train_pd[(~train_pd['Id'].isin(test_ids))]

def eliminate_features():
    global features
    global useless_features

    features = [x for x in features if x not in useless_features]
    print(features)
    print("features:", len(features))

    test_rmse, _, _, _ = predict_sub_set(test_data, general_train_fold, lgbm_params)
    print("test rmse:", test_rmse)

    zero_importance_dict = {}
    importance_dict = {}
    RMSEs = []
    for fold in range(0, 5):
        fold_path = f'data_splits/Fold_{fold}.txt'
        fold_ids = get_test_indexes_from_file(fold_path)

        val_fold = general_train_fold.loc[train_pd['Id'].isin(fold_ids)].copy()
        train_fold = general_train_fold.loc[(~train_pd['Id'].isin(fold_ids))].copy()

        rmse, model, X_val, y_val = predict_sub_set(val_fold, train_fold, lgbm_params)
        RMSEs.append(rmse)

        result = permutation_importance(model, X_val, y_val, scoring='neg_root_mean_squared_error', n_repeats=3)

        print(
            f"{Fore.GREEN}{Style.BRIGHT}Important features: {(result['importances_mean'] > 0).mean():.0%}   ({rmse=:.3f}){Style.RESET_ALL}")
        importance_df = pd.DataFrame({'feature': features, 'importance': result['importances_mean'],
                                      'std': result['importances_std']}, index=X_val.columns).sort_values('importance',
                                                                                                         ascending=False)
        importance_df['ComputationTypeId'] = concepts.set_index('Name').ComputationTypeId
        importance_df.fillna({'ComputationTypeId': 'Player'}, inplace=True)

        important_features = importance_df[importance_df['importance'] > 0]
        important_features_list = list(important_features['feature'].unique())
        for imp in important_features_list:
            importance_dict[imp] = importance_dict.get(imp, 0) + importance_df[importance_df['feature'] == imp]['importance'].values[0]

        print(importance_df.shape)
        zero_importance = importance_df[importance_df['importance'] <= 0]
        print("zero importance:", zero_importance.shape)
        zero_imp_list = list(zero_importance['feature'].unique())

        for zi in zero_imp_list:
            zero_importance_dict[zi] = zero_importance_dict.get(zi, 0) + 1

        print()

    print(features)
    print("Mean RMSE:", np.mean(RMSEs), "Std RMSE:", np.std(RMSEs), "Test RMSE:", test_rmse, "features:", len(features))

    with open('experiments.txt', 'a') as f:
        # Write the list of items, separated by commas
        f.write("Feature List: " + ', '.join(features) + '\n')
        f.write("Params lgb: " + ', '.join([f"{key}: {value}" for key, value in lgbm_params.items()]) + '\n')
        # Write the additional results like RMSE and feature count
        f.write(f"Mean RMSE: {np.mean(RMSEs):.4f}, Std RMSE: {np.std(RMSEs):.4f} ,Test RMSE: {test_rmse:.4f}, Features: {len(features)}\n")
        f.write('-' * 80 + '\n')  # Add a separator for readability

    useless_features = []
    for key in zero_importance_dict:
        if zero_importance_dict[key] >= 4:
            useless_features.append(key)

    print(useless_features)

    print("=========================================")
    return features, useless_features, importance_dict


useless_features = []
while True:
    features, useless_features, importance_dict = eliminate_features()
    if len(useless_features) == 0:
        sorted_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        for key in sorted_dict:
            print(key, ":", sorted_dict[key])
        break

