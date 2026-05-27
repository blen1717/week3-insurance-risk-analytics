import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

class HypothesisTester:
    def init(self, df, alpha=0.05):
        self.df = df
        self.alpha = alpha
        self.results = []

    def test_two_proportions(self, group_col, group_a, group_b, target='ClaimStatus'):
        """Z-test for difference in claim rates between two groups"""
        a = self.df[self.df[group_col] == group_a]
        b = self.df[self.df[group_col] == group_b]
        count = [a[target].sum(), b[target].sum()]
        nobs = [len(a), len(b)]
        z_stat, p_value = proportions_ztest(count, nobs)
        return {
            'hypothesis': f"{group_col}: {group_a} vs {group_b}",
            'p_value': p_value,
            'reject_null': p_value < self.alpha,
            'group_a_rate': count[0]/nobs[0],
            'group_b_rate': count[1]/nobs[1]
        }

    def test_anova(self, group_col, target='LossRatio'):
        """ANOVA to compare means across multiple groups"""
        groups = [self.df[self.df[group_col] == g][target].dropna() for g in self.df[group_col].unique()]
        f_stat, p_value = stats.f_oneway(*groups)
        return {
            'hypothesis': f"{group_col} differences in {target}",
            'p_value': p_value,
            'reject_null': p_value < self.alpha,
            'f_statistic': f_stat
        }

    def test_t_test(self, group_col, group_a, group_b, target='Margin'):
        """Welch's t-test for continuous variable between two groups"""
        a = self.df[self.df[group_col] == group_a][target]
        b = self.df[self.df[group_col] == group_b][target]
        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
        return {
            'hypothesis': f"{group_col}: {group_a} vs {group_b} ({target})",
            'p_value': p_value,
            'reject_null': p_value < self.alpha,
            'effect_size': a.mean() - b.mean(),
            'group_a_mean': a.mean(),
            'group_b_mean': b.mean()
        }
