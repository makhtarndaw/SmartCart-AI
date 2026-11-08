# SmartCart AI Streamlit prototype

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Connect the trained Colab model
In Colab:
```python
import joblib
joblib.dump(best_xgb_model, "best_xgb_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
```
Download both files and place them in `models/`.

## Important
- Built-in demo data are synthetic and only demonstrate the interface.
- The trained model was built on REES46 cosmetics data. Other companies' data must be mapped to matching feature definitions and distributions, or the model must be retrained.
- Marketing actions are suggestions to test, not proven causal effects.
- Campaign measurement requires actual campaign exposure and outcome data.
