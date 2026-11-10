"""Outcome-availability splits, paired uncertainty and diagnostic summaries."""
import numpy as np
import pandas as pd
from .modeling import ForecastModel


def temporal_split(data, origin, end, maturity_days=0, score_asof=None):
    origin, end = pd.Timestamp(origin), pd.Timestamp(end)
    known = data.valid_target & (data.construction_end_date < origin)
    if maturity_days:
        known &= data.construction_start_date < origin-pd.Timedelta(days=maturity_days)
    period = data.construction_start_date.between(origin, end)
    observed = period & data.valid_target
    if score_asof is not None:
        observed &= data.construction_end_date < pd.Timestamp(score_asof)
    train, test = data.loc[known].copy(), data.loc[observed].copy()
    assert len(train) and len(test)
    assert train.construction_end_date.max() < origin
    assert set(train.home_id).isdisjoint(test.home_id)
    audit = {'origin': origin, 'end': end, 'train_n': len(train),
             'cohort_n': int(period.sum()), 'scored_n': len(test),
             'not_scored': int(period.sum()-len(test)),
             'scored_pct': 100*len(test)/period.sum(), 'maturity_days': maturity_days}
    return train, test, audit


def metrics(y, p):
    e = np.asarray(p)-np.asarray(y)
    return {'MAE_days': np.abs(e).mean(), 'RMSE_days': np.sqrt((e**2).mean()),
            'bias_days': e.mean(), 'p90_abs_error': np.quantile(np.abs(e), .9),
            'within_30_pct': 100*(np.abs(e)<=30).mean()}


def backtest(data, windows, specs, maturity_days=0, score_asof=None):
    results, predictions, audits = [], [], []
    for fold, origin, end in windows:
        train, test, audit = temporal_split(data, origin, end, maturity_days, score_asof)
        audits.append(dict(fold=fold, **audit))
        keep = ['home_id','construction_start_date','cycle_time_days','site_manager',
                'plan_code','community_clean','metro','sqft','manager_starts_90d','start_sin',
                'is_spec_home','product_line']
        p = test[keep].copy()
        p['fold'] = fold
        for name, (kind, columns) in specs.items():
            fitted = ForecastModel(columns, kind).fit(train)
            p[name] = fitted.predict(test)
            results.append(dict(fold=fold, model=name, n=len(test),
                                **metrics(test.cycle_time_days, p[name])))
        predictions.append(p)
    return pd.DataFrame(results), pd.concat(predictions, ignore_index=True), pd.DataFrame(audits)


def block_mean_interval(values, blocks, draws=2000, seed=42, level=.95):
    """Resample calendar-month blocks; conditional on already fitted models.

    Retains all homes in each sampled block. This handles within-block dependence,
    not dependence across adjacent months, model fitting or assignment uncertainty.
    """
    d = pd.DataFrame({'value': np.asarray(values), 'block': np.asarray(blocks)})
    g = d.groupby('block').value.agg(['sum','count'])
    if len(g) < 2:
        return float(d.value.mean()), np.nan, np.nan, len(g)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(g), size=(draws, len(g)))
    estimates = g['sum'].to_numpy()[ix].sum(1)/g['count'].to_numpy()[ix].sum(1)
    a = (1-level)/2
    return float(d.value.mean()), *np.quantile(estimates, [a,1-a]), len(g)


def manager_summary(oof, expected_column, alternate_column):
    d = oof.copy()
    d['residual'] = d.cycle_time_days-d[expected_column]
    d['alt_residual'] = d.cycle_time_days-d[alternate_column]
    # Remove common forecast bias per evaluation period, not manager-specific errors.
    for c in ['residual','alt_residual']:
        d[c] -= d.groupby('fold')[c].transform('mean')
    d['month'] = d.construction_start_date.dt.to_period('M').astype(str)
    rows=[]
    for manager, g in d.groupby('site_manager'):
        avg, lo, hi, blocks = block_mean_interval(g.residual, g.month)
        rows.append({'site_manager':manager, 'n':len(g), 'months':blocks,
            'raw_days':g.cycle_time_days.mean(), 'adjusted_days':avg, 'lo95':lo,'hi95':hi,
            'alternative_days':g.alt_residual.mean(), 'communities':g.community_clean.nunique(),
            'plans':g.plan_code.nunique()})
    result=pd.DataFrame(rows).set_index('site_manager')
    result['raw_rank']=result.raw_days.rank()
    result['adjusted_rank']=result.adjusted_days.rank()
    return result.sort_values('adjusted_days'), d


def segment_errors(frame, model, baseline, group):
    d=frame.copy()
    d['abs_error']=(d[model]-d.cycle_time_days).abs()
    d['baseline_error']=(d[baseline]-d.cycle_time_days).abs()
    d['error']=d[model]-d.cycle_time_days
    return d.groupby(group, observed=True).agg(n=('home_id','size'), MAE_days=('abs_error','mean'),
        baseline_MAE=('baseline_error','mean'),bias_days=('error','mean'),
        p90_abs_error=('abs_error',lambda s:s.quantile(.9)))
