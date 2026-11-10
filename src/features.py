"""Small, motivated feature families; all inputs restricted to start-day state."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SPEC = ['plan_code', 'product_line', 'beds', 'baths', 'half_baths', 'sqft',
        'stories', 'garage_size_clean', 'basement_type_clean', 'lot_type', 'lot_size_sqft', 'is_spec_home']
GEO = ['community_clean', 'metro', 'permit_authority']
TIME = ['start_sin', 'start_cos', 'start_year', 'permit_days', 'permit_to_start_days',
        'sale_lead_days', 'sale_recorded']
PRESSURE = ['manager_starts_90d', 'community_starts_90d']
FAMILIES = {'M2_spec': SPEC, 'M3_geo': SPEC+GEO, 'M4_time': SPEC+GEO+TIME,
            'M5_pressure': SPEC+GEO+TIME+PRESSURE,
            'M6_manager': SPEC+GEO+TIME+PRESSURE+['site_manager']}


def start_features(homes, event_roster):
    """Counts prior starts in [t-90d,t); no end dates, targets, or same-day peers.

    Roster may include later starts: search boundaries guarantee they cannot affect
    earlier features. Assumes the exports cover the builder's relevant start ledger.
    Manager/community values are assumed to be the assignments at each start.
    """
    d = homes.copy()
    t = d.construction_start_date
    d['start_sin'] = np.sin(2*np.pi*(t.dt.month-1)/12)
    d['start_cos'] = np.cos(2*np.pi*(t.dt.month-1)/12)
    d['start_year'] = t.dt.year.astype(float)
    d['permit_days'] = (d.permit_issue_date-d.permit_application_date).dt.days.where(
        (d.permit_issue_date <= t) & (d.permit_application_date <= d.permit_issue_date))
    d['permit_to_start_days'] = (t-d.permit_issue_date).dt.days.where(d.permit_issue_date <= t)
    d['sale_lead_days'] = (t-d.sale_date).dt.days.where(d.sale_date <= t)
    d['sale_recorded'] = (d.sale_date.notna() & (d.sale_date <= t)).astype(int)
    events = event_roster.drop_duplicates('home_id')
    for key, out in [('site_manager', PRESSURE[0]), ('community_clean', PRESSURE[1])]:
        counts = pd.Series(0., index=d.index)
        for group, rows in d.groupby(key, dropna=False):
            dates = np.sort(events.loc[events[key].eq(group), 'construction_start_date'].to_numpy())
            query = rows.construction_start_date.to_numpy()
            counts.loc[rows.index] = (np.searchsorted(dates, query, side='left') -
                np.searchsorted(dates, query-np.timedelta64(90, 'D'), side='left'))
        d[out] = counts
    return d


class SqftRepair(BaseEstimator, TransformerMixin):
    """Fit plan references on training covariates only; never use holdout medians.

    Factor-of-ten is a supported data-entry hypothesis, not external confirmation.
    Only repair when dividing by ten lands within 20% of the fitted plan median.
    Unknown plans and unsupported anomalies are left unchanged.
    """
    def fit(self, X, y=None):
        self.medians_ = X.groupby('plan_code').sqft.median()
        return self

    def transform(self, X):
        d = X.copy()
        ref = d.plan_code.map(self.medians_)
        fix = (d.sqft > 3*ref) & ((d.sqft/10/ref).between(.8, 1.2))
        d['sqft'] = d.sqft.astype(float).where(~fix, d.sqft/10)
        return d
