"""Two baselines, an additive reference and one nonlinear tabular model."""
import numpy as np
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .features import SqftRepair


class ForecastModel:
    def __init__(self, columns=None, kind='catboost', seed=42):
        self.columns = columns
        self.kind = kind
        self.seed = seed

    def fit(self, data):
        y = data.cycle_time_days.to_numpy()
        self.global_mean_ = y.mean()
        self.plan_means_ = data.groupby('plan_code').cycle_time_days.mean()
        if self.kind in ('global', 'plan'):
            return self
        self.repair_ = SqftRepair().fit(data)
        x = self.repair_.transform(data)[self.columns].copy()
        self.cats_ = x.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        x = self._prepare(x)
        if self.kind == 'ridge':
            num = [c for c in self.columns if c not in self.cats_]
            prep = ColumnTransformer([
                ('num', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)),
                                  ('scale', StandardScaler())]), num),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.cats_)])
            self.model_ = Pipeline([('prep', prep), ('reg', Ridge(alpha=20, solver='lsqr'))])
        elif self.kind == 'catboost':
            self.model_ = CatBoostRegressor(iterations=450, depth=5, learning_rate=.045,
                loss_function='RMSE', l2_leaf_reg=8, random_seed=self.seed,
                thread_count=2, verbose=False, allow_writing_files=False, cat_features=self.cats_)
        else:
            raise ValueError(self.kind)
        self.model_.fit(x, y)
        return self

    def _prepare(self, x):
        x = x.copy()
        for c in x:
            if c in self.cats_:
                x[c] = x[c].fillna('Unknown').astype(str)
            else:
                x[c] = x[c].astype(float)
        return x

    def predict(self, data):
        if self.kind == 'global':
            p = np.full(len(data), self.global_mean_)
        elif self.kind == 'plan':
            p = data.plan_code.map(self.plan_means_).fillna(self.global_mean_).to_numpy()
        else:
            p = self.model_.predict(self._prepare(self.repair_.transform(data)[self.columns]))
        return np.maximum(np.asarray(p, dtype=float), 1.)
