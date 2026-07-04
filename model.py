from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

def create_pipeline(preprocessor):
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        eval_metric='logloss',
        random_state=42
    )
    
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb_model)
    ])
    
    return full_pipeline
