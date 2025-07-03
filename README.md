# HealthAI

This repository contains a Flask application for predicting health outcomes.

The application requires machine learning models stored in the `models/` directory.
These models are not tracked in version control and must be generated locally by
running:

```bash
python train_and_export_models.py
```

This script trains the models and exports the resulting `*.pkl` files and
`feature_info.json` into the `models/` folder. Run it before starting the
Flask app.
