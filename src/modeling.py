import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score

class RiskModeler:
    def init(self, model_type='severity'):
        self.model_type = model_type
        self.results = {}
    
    def train_models(self, X_train, y_train, X_test, y_test):
        if self.model_type == 'severity':
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
                'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
            }
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                self.results[name] = {'RMSE': rmse, 'R2': r2, 'model': model}
        else:  # classification
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
                'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
            }
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0
                self.results[name] = {'Accuracy': acc, 'AUC': auc, 'model': model}
        
        return pd.DataFrame(self.results).T
    
    def get_best_model(self):
        if self.model_type == 'severity':
            best = max(self.results.items(), key=lambda x: x[1]['R2'])
        else:
            best = max(self.results.items(), key=lambda x: x[1]['AUC'])
        return best[1]['model'], best[0]
