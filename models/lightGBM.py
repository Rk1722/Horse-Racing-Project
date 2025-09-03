#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


df = pd.read_csv("/Users/riyaazkhan/Documents/Imperial_Maths/Algorithmic_Trading_Club/Projects/Horse-Racing-Project/Data/Processed/XGBoost_Dataset.v2_trimmed.csv")


# In[12]:


#df2 = df.copy()


# In[10]:


#Maybe include margin, and res_place in the racers as a feature


# In[5]:


from sklearn.preprocessing import LabelEncoder

# Example: your dataframe has a column 'country_code'
df['countryCode'] = df['countryCode'].astype('category')
df['headGear'] = df['headGear'].astype('category')
df['prev_headgear'] = df['prev_headgear'].astype('category')
df['course_clean'] = df['course_clean'].astype('category')
df['course_surface'] = df['course_surface'].astype('category')
df['track_layout'] = df['track_layout'].astype('category')
df['race_jump_type'] = df['race_jump_type'].astype('category')
df['best_dist_band_hist'] = df['best_dist_band_hist'].astype('category')


# In[9]:


unnecessarycolumns = ['rid',
    'horseName',
    'age',
    'saddle',
    'isFav',
                      'trainerName','jockeyName','position','positionL','dist',
                      'weightSt','weightLb','RPR','TR','OR','father','mother',
                      'gfather','runners','margin','res_place','time','band_bins',
                      'condition','hurdles','rclass','course',
                      'prizes','winningTime','metric','source_year',
                      'prev_raceId','close_finish','date2','title','saddle_rank_bin',
                     'band_clean','band','ages',
    'days_since_last_race_binned',
    'distance_banding','decimalPrice',
                      'race_age_band','previous_trainer','previous_jockey']

df = df.drop(columns=unnecessarycolumns)


# In[18]:


tr_data = df[df['year'] <= 2014]
val_data = df[(df['year'] < 2018) & (df['year'] > 2014)]
bkt_data = df[df['year'] >= 2018]


# In[19]:


tr_data = tr_data.copy().sort_values('date')
val_data = val_data.copy().sort_values('date')
bkt_data = bkt_data.copy().sort_values('date')

tr_data = tr_data.drop(columns='date')
val_data = val_data.drop(columns='date')
bkt_data = bkt_data.drop(columns='date')


# In[20]:


tr_data_y = tr_data['res_win']
tr_data_p = tr_data['betting_prob_win']
tr_data_x = tr_data.drop(columns=['res_win','betting_prob_win'])

val_data_y = val_data['res_win']
val_data_p = val_data['betting_prob_win']
val_data_x = val_data.drop(columns=['res_win','betting_prob_win'])

bkt_data_y = bkt_data['res_win']
bkt_data_p = bkt_data['betting_prob_win']
bkt_data_x = bkt_data.drop(columns=['res_win','betting_prob_win'])


# In[21]:


import lightgbm as lgb

modelv1 = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=63,
    min_data_in_leaf=200,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs =-1,
)


# In[22]:


modelv1.fit(
    tr_data_x, tr_data_y,
    eval_set=[(val_data_x, val_data_y)],
    eval_metric=['logloss','auc'],                    # or 'auc'
    categorical_feature=[c for c in tr_data_x.columns if str(tr_data_x[c].dtype)=='category'],
    callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(100)]
)


# In[25]:


val_preds = modelv1.predict_proba(val_data_x, num_iteration=modelv1.best_iteration_)[:, 1]


# In[26]:


from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
print("LogLoss:", log_loss(val_data_y, val_preds))
print("AUC:", roc_auc_score(val_data_y, val_preds))
print("Brier:", brier_score_loss(val_data_y, val_preds))


# In[27]:


bkt_preds = modelv1.predict_proba(bkt_data_x,num_iteration=modelv1.best_iteration_)[:, 1]


# In[28]:


print("LogLoss:", log_loss(bkt_data_y, bkt_preds))
print("AUC:", roc_auc_score(bkt_data_y, bkt_preds))
print("Brier:", brier_score_loss(bkt_data_y, bkt_preds))


# In[40]:


import numpy as np
import pandas as pd

def backtesting_strategy(pred_probs, market_probs, results, stake=1.0, min_edge=0.0,overround = 0.25):
    """
    pred_probs   : pd.Series or np.ndarray of model win probabilities in [0,1]
    market_probs : pd.Series or np.ndarray of market/implied win probabilities in [0,1]
    results      : pd.Series or np.ndarray of 0/1 race outcomes (1 = win)
    stake        : fixed stake per bet (default £1)
    min_edge     : only bet if (pred - market) > min_edge (edge filter), default 0.0
    """
    # to aligned DataFrame (keeps your original index if Series)
    total_number_of_horses = len(pred_probs)
    df = pd.DataFrame({
        'pred': np.asarray(pred_probs, dtype=float),
        'mkt' : np.asarray(market_probs, dtype=float),
        'res' : np.asarray(results, dtype=float)
    })

    # basic cleaning / clipping to avoid div-by-zero or >1
    df['pred'] = df['pred'].clip(0.0, 1.0)
    df['mkt']  = df['mkt'].clip(1e-9, 1.0 - 1e-9)  # can't be 0 or 1 for odds
    df['bet']  = (df['pred'] - df['mkt']) > min_edge

    # fair decimal odds from market prob (replace with real odds if you have them)
    df['odds'] = (1.0 / df['mkt'])/(1 + overround)

    # PnL rules
    win_profit = stake * (df['odds'] - 1.0)  # net profit when win
    lose_profit = -stake                     # lose stake when lose

    df['PnL'] = 0.0
    df.loc[df['bet'] & (df['res'] == 1), 'PnL'] = win_profit[df['bet'] & (df['res'] == 1)]
    df.loc[df['bet'] & (df['res'] == 0), 'PnL'] = lose_profit

    # cumulative metrics
    df['cum_PnL'] = df['PnL'].cumsum()
    df['n_bets'] = df['bet'].cumsum()

    # summary
    n_bets = int(df['bet'].sum())
    total_staked = n_bets * stake
    total_pnl = df['PnL'].sum()
    hit_rate = (df.loc[df['bet'], 'res'].mean() if n_bets > 0 else np.nan)
    avg_edge = (df.loc[df['bet'], 'pred'].sub(df.loc[df['bet'], 'mkt']).mean()
                if n_bets > 0 else np.nan)
    roi_per_bet = (total_pnl / total_staked) if total_staked > 0 else np.nan

    summary = {
        'bets': n_bets,
        'total_staked': total_staked,
        'total_PnL': total_pnl,
        'ROI_per_bet': roi_per_bet,
        'hit_rate_on_bets': hit_rate,
        'avg_edge_on_bets': avg_edge,
        'total_possible_bets': total_number_of_horses
    }

    return df, summary


# In[41]:


bkt_df, summary = backtesting_strategy(
    pred_probs=bkt_preds,          # your model probs
    market_probs=bkt_data_p,      # your market/implied probs
    results=bkt_data_y,                 # 0/1 results
    stake=1.0,
    min_edge=0.00                       # set e.g. 0.02 to only bet with ≥2% edge
)


# In[42]:


print(summary)


# In[51]:


import pandas as pd
import numpy as np

# Align indices just in case
common_idx = tr_data_x.index.intersection(tr_data_p.index)
X = tr_data_x.loc[common_idx]
p = pd.Series(tr_data_p, index=common_idx).astype(float)

# Keep only numeric columns (coerce if needed)
X_num = X.apply(pd.to_numeric, errors='coerce')

# Drop columns that are all NaN or constant (no variance)
is_const = X_num.nunique(dropna=True) <= 1
X_num = X_num.loc[:, ~is_const]

# Compute Pearson correlations (row-wise NaNs handled automatically)
corr_s = X_num.corrwith(p)

# Show strongest correlations
corr_df = (corr_s.dropna()
                 .to_frame('corr')
                 .assign(abs_corr=lambda d: d['corr'].abs())
                 .sort_values('abs_corr', ascending=False))

print(corr_df.head(30))


# In[ ]:




