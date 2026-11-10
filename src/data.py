"""Explicit CSV semantics and deterministic cleaning (no learned statistics)."""
from pathlib import Path
import numpy as np
import pandas as pd

DATES = ['sale_date', 'permit_application_date', 'permit_issue_date',
         'construction_start_date', 'construction_end_date', 'final_inspection_date']


def locate_data(root):
    """Support the supplied project layout and this attachment workspace."""
    root = Path(root)
    standard = (root / 'data/build_history.csv', root / 'data/homes_to_predict.csv')
    attached = (root / 'project_sources/03-build_history.csv',
                root / 'project_sources/02-homes_to_predict.csv')
    for pair in (standard, attached):
        if all(p.is_file() for p in pair):
            return pair
    raise FileNotFoundError('Put build_history.csv and homes_to_predict.csv in data/.')


def read_export(path):
    # Pandas otherwise interprets the valid basement category "None" as NA.
    return pd.read_csv(path, keep_default_na=False, na_values=[''])


def clean_export(raw):
    d = raw.drop_duplicates().copy().reset_index(drop=True)
    if d.home_id.isna().any() or d.home_id.duplicated().any():
        raise ValueError('Missing or conflicting duplicate home IDs require review.')
    for col in DATES:
        parsed = pd.to_datetime(d[col], errors='coerce')
        if (d[col].notna() & parsed.isna()).any():
            raise ValueError(f'Unparseable non-empty dates in {col}')
        d[col] = parsed
    if d.construction_start_date.isna().any():
        raise ValueError('Construction start is required.')
    d['community_clean'] = d.community.str.strip().str.lower()
    d['basement_type_clean'] = d.basement_type.fillna('Unknown').str.strip()
    garage = {'1': 1, 'single': 1, '2': 2, 'double': 2, '3': 3, 'triple': 3}
    d['garage_size_clean'] = d.garage_size.astype(str).str.strip().str.lower().map(garage)
    if d.garage_size.notna().any() and d.garage_size_clean.isna().any():
        raise ValueError('Unrecognized garage encoding.')
    d['cycle_time_days'] = (d.construction_end_date - d.construction_start_date).dt.days
    d['has_observed_target'] = d.construction_end_date.notna()
    d['valid_target'] = d.cycle_time_days.gt(0)
    return d


def cohort_audit(history):
    return (history.assign(start_quarter=history.construction_start_date.dt.to_period('Q'))
            .groupby('start_quarter').agg(starts=('home_id', 'size'),
             observed=('has_observed_target', 'sum'), valid=('valid_target', 'sum'),
             mean_observed_days=('cycle_time_days', lambda s: s[s > 0].mean()))
            .assign(valid_pct=lambda d: 100*d.valid/d.starts))


def assert_prediction_contract(predictions, homes):
    assert list(predictions.columns) == ['home_id', 'predicted_cycle_days']
    assert len(predictions) == len(homes) == 578
    assert predictions.home_id.tolist() == homes.home_id.tolist()
    assert predictions.home_id.is_unique
    assert np.isfinite(predictions.predicted_cycle_days).all()
    assert predictions.predicted_cycle_days.gt(0).all()
