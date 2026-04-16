
import joblib
from pathlib import Path

def train_linear_model(model_object, X_train, y_train, **kwargs):
    if kwargs:
        model_object.set_params(**kwargs)
        
    model_name = model_object.__class__.__name__
    trained_model = model_object.fit(X_train, y_train)

    base_dir = Path(__file__).resolve().parent.parent 
    models_dir = base_dir / "models"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = models_dir / f"{model_name}.joblib"
    joblib.dump(trained_model, save_path)
    
    return trained_model