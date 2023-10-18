import os
import itertools
import pandas as pd
import numpy as np
from datetime import date
from tqdm.notebook import tqdm
from workalendar.europe import Russia
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import KFold
from src.helpers import get_project_dir


def group(df):
    df2 = df.groupby(['point', 'date']) \
            .agg({'likescount': ['sum', 'count'],
                  'commentscount': 'sum',
                  'symbols_cnt': 'sum','words_cnt':
                  'sum','hashtags_cnt': 'sum',
                  'mentions_cnt': 'sum',
                  'links_cnt': 'sum',
                  'emoji_cnt': 'sum',
                  'lon': 'mean',
                  'lat': 'mean'}) \
            .reset_index()
    df2.columns = ['_'.join(col) for col in df2.columns.values]
    df2 = df2.rename(columns={'likescount_count': 'count', 'point_': 'point', 'date_': 'date'})
    return df2


def custom_score(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred)/y_pred)


def eval(model, df, df_v, df_t, feats, target, is_fitted=False):
    if not is_fitted:
        model.fit(df[feats], df[target])
    pred = model.predict(df[feats])
    scores = {}
    scores['train R2'] = r2_score(y_true=df[target], y_pred=pred)
    scores['train custom'] = custom_score(y_true=df[target], y_pred=pred)
    scores['train MAPE'] = mean_absolute_percentage_error(y_true=df[target], y_pred=pred)
    print(f"train R2 = {scores['train R2']}")
    print(f"train custom = {scores['train custom']}")
    print(f"train MAPE = {scores['train MAPE']}")
    pred = model.predict(df_v[feats])
    scores['val R2'] = r2_score(y_true=df_v[target], y_pred=pred)
    scores['val custom'] = custom_score(y_true=df_v[target], y_pred=pred)
    scores['val MAPE'] = mean_absolute_percentage_error(y_true=df_v[target], y_pred=pred)
    print(f"val R2 = {scores['val R2']}")
    print(f"val custom = {scores['val custom']}")
    print(f"val MAPE = {scores['val MAPE']}")
    pred = model.predict(df_t[feats])
    scores['test R2'] = r2_score(y_true=df_t[target], y_pred=pred)
    scores['test custom'] = custom_score(y_true=df_t[target], y_pred=pred)
    scores['test MAPE'] = mean_absolute_percentage_error(y_true=df_t[target], y_pred=pred)
    print(f"test R2 = {scores['test R2']}")
    print(f"test custom = {scores['test custom']}")
    print(f"test MAPE = {scores['test MAPE']}")
    return scores


def cross_validate_(model, params, X, y, cv, scoring, random_state=42):
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = dict([(s, []) for s in scoring.keys()])
    for i, (train_index, test_index) in enumerate(tqdm(kf.split(X), total=cv)):
        y_pred = model(**params).fit(X.loc[train_index],
                                     y.loc[train_index]) \
                                .predict(X.loc[test_index])
        for s in scores.keys():
            scores[s].append(scoring[s](y_true=y.loc[test_index],
                                        y_pred=y_pred))
    for s in scores.keys():
        print(f"train cv {s} = {np.mean(scores[s])}")
    return scores



class Selector:

    def search(self, df, df_v, feats_to_select, default_feats, target, model, params, scorer, init_score=100,
               initial_direction='forward', max_steps=2):
        direction = initial_direction
        best_score = init_score
        steps = 0
        best_feats = []
        print('='*100)
        if initial_direction == 'forward':
            print('Forward')
            current_feats = []
            last_turn = 0
        if initial_direction == 'backward':
            print('Backward')
            current_feats = feats_to_select.copy()
            last_turn = len(feats_to_select)
        while True:
            print('-'*70)
            print(f'last_turn: {last_turn}, best_score: {best_score}, '
                  f'best_feats_cnt: {len(best_feats)}, curr_feats: {current_feats}')
            if direction == 'forward':
                feat, score = self.forward(df, df_v, current_feats, feats_to_select, default_feats, target, model, params, scorer)
                print(f'new_feat: {feat}, score: {score}')
                current_feats.append(feat)
                if score <= best_score:
                    print(f'UPDATING best set\nbest_feats: {current_feats}, best_score: {score}')
                    best_score = score
                    best_feats = current_feats.copy()
                    steps = 0
                else:
                    print(f'No improvement')
                    steps += 1
                if steps == max_steps:
                    if last_turn != len(best_feats):
                        print('='*100)
                        print(f'No improvement for {max_steps} steps. -> Backward')
                        direction = 'backward'
                        current_feats = best_feats.copy()
                        last_turn = len(best_feats)
                        steps = 0
                    else:
                        print('=' * 100)
                        print(f'No improvement for {max_steps} steps. -> End')
                        break
                else:
                    if (len(current_feats) == len(feats_to_select)):
                        if last_turn != len(best_feats):
                            print('='*50)
                            print(f'Out of features. -> Backward')
                            direction = 'backward'
                            current_feats = best_feats.copy()
                            last_turn = len(best_feats)
                            steps = 0
                        else:
                            print('='*100)
                            print(f'Out of features. -> End')
                            break
            else:
                feat, score = self.backward(df, df_v, current_feats, feats_to_select, default_feats, target, model, params, scorer)
                print(f'feat_to_exclude: {feat}, score: {score}')
                current_feats.remove(feat)
                if score <= best_score:
                    print(f'Updating best set\nbest_feats: {current_feats}, best_score: {best_score}')
                    best_score = score
                    best_feats = current_feats.copy()
                    steps = 0
                else:
                    print(f'No improvement')
                    steps += 1
                if steps == max_steps:
                    if last_turn != len(best_feats):
                        print('='*100)
                        print(f'No improvement for {max_steps} steps. -> Forward')
                        direction = 'forward'
                        current_feats = best_feats.copy()
                        last_turn = len(best_feats)
                        steps = 0
                    else:
                        print('=' * 100)
                        print(f'No improvement for {max_steps} steps. -> End')
                        break
                else:
                     if len(current_feats) == 0:
                        if last_turn != len(best_feats):
                            print('='*100)
                            print('No features left to exclude. -> Forward')
                            direction = 'forward'
                            current_feats = best_feats.copy()
                            last_turn = len(best_feats)
                            steps = 0
                        else:
                            print('='*100)
                            print('No features left to exclude. -> End')
                            break
        return best_feats, best_score

    def forward(self, df, df_v, current_feats, feats_to_select, default_feats, target, model, params, scorer):
        left_feats = [k for k in feats_to_select if k not in current_feats]
        scores = []
        for feat in left_feats:
            model_ = model(**params)
            model_.fit(df[default_feats+current_feats+[feat]], df[target])
            pred = model_.predict(df_v[default_feats+current_feats+[feat]])
            scores.append(scorer(y_true=df_v[target], y_pred=pred))
        best_score = min(scores)
        new_feat = left_feats[scores.index(best_score)]
        return new_feat, best_score


    def backward(self, df, df_v, current_feats, feats_to_select, default_feats, target, model, params, scorer):
        scores = []
        for feat in current_feats:
            curr_fts = [a for a in current_feats if a!=feat]
            model_ = model(**params)
            model_.fit(df[default_feats+curr_fts], df[target])
            pred = model_.predict(df_v[default_feats+curr_fts])
            scores.append(scorer(y_true=df_v[target], y_pred=pred))
        best_score = min(scores)
        feat_to_exclude = current_feats[scores.index(best_score)]
        return feat_to_exclude, best_score


def grid_search(df, df_v, feats, target, model, params, scorer, default_params={}):
    combinations = [dict(zip(params.keys(), v)) for v in itertools.product(*params.values())]
    iterator = tqdm(combinations, total=len(combinations))
    iterator.set_postfix({'best_score': None, 'best_set': None})
    best_score = None
    best_set = None
    for it, comb in enumerate(iterator):
        print(f'Iteration: {it}, Set: {comb}', end=' ')
        model_ = model(**comb, **default_params)
        model_.fit(df[feats], df[target])
        pred = model_.predict(df_v[feats])
        score = scorer(y_true=df_v[target], y_pred=pred)
        print(f'Score: {score}')
        if it == 0:
            best_score = score
            best_set = comb.copy()
        elif score < best_score:
            best_score = score
            best_set = comb.copy()
        iterator.set_postfix({'best_score': best_score, 'best_set': best_set})
    return best_set, best_score


COUNTRY = Russia()
HOLIDAYS = [item[0] for item in COUNTRY.holidays(2019)] + [item[0] for item in COUNTRY.holidays(2020)]
WEATHER = pd.read_excel(os.path.join(get_project_dir(), 'data/weather_spb.xlsx'))
WEATHER = dict([(a, b) for (a,b) in zip(WEATHER['date'], WEATHER['day_temp'])])
def get_new_feats(x):
    output_ = [x.hour,
               x.day,
               x.month,
               x.year,
              1 if (x.hour >= 23) and (x.hour < 7) else 0,
              1 if (x.hour >= 19) and (x.hour < 23) else 0,
              1 if (x.hour >= 7) and (x.hour < 9) else 0,
              1 if (x.hour >= 9) and (x.hour < 19) else 0,
              x.weekday(),
              1 if x.weekday() in [5, 6] else 0,
              1 if date(x.year, x.month, x.day) in HOLIDAYS else 0,
              1 if COUNTRY.is_working_day(date(x.year, x.month, x.day)) else 0,
              WEATHER[pd.Timestamp(year=x.year, month=x.month, day=x.day)]]
    return output_