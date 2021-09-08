# Configs files

- default_france.yaml: configurations to run the French DCD model (switch on/off features, xgboost parameters, etc.)
- default_trainer_france.yaml: configs used by main_trainer.py script to train sequentially models
(e.g. first date_when_predicting, last date_to_predict, xgboost parameters, etc.)
- longterm_default.yaml: configs file for long-term model using Facebook Prophet
- default_future_forecasts_france.yaml: configs for future forecasts

Those files can be used as reference to run on other countries

# Objects
- Configs: object for single run (train and/or predict)
- ConfigsTrainer: object for multiple models trainer (train only)
- FeatureController: object to control which features should be switched on or off in final model (train & predict)
