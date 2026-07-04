from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import preprocessing
import model

def main():
    print("1. Loading data...")
    df = preprocessing.get_raw_data()
    
    X = df.drop(columns=['churned'])
    y = df['churned']
    
    print("Splitting data into Train/Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    print(" Initializing preprocessing pipeline...")
    preprocessor = preprocessing.build_preprocessor()
    
    print("Building final XGBoost pipeline")
    clf_pipeline = model.create_pipeline(preprocessor)
    
    print("Model training")
    clf_pipeline.fit(X_train, y_train)
    
    print("Making predictions and evaluating")
    y_pred = clf_pipeline.predict(X_test)
    y_pred_proba = clf_pipeline.predict_proba(X_test)[:, 1]
    
    print("\n=== EXECUTION SUCCESSFUL ===")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()