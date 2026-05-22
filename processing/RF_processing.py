import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from utils.metrics import error_metrics


class RFProcessor:
    def __init__(self, name, params, predict_config=None):
        self.name = name
        self.params = params
        self.predict_config = predict_config
        self.original_dir = os.getcwd()

    def train(self, X_train, y_train, X_val, y_val, importance=True, features=None, target=None):
        """
        Trains the Random Forest model and optionally computes feature importance.

        Parameters:
        - X_train (DataFrame): Training features.
        - y_train (DataFrame): Training target.
        - X_val (DataFrame): Validation features.
        - y_val (DataFrame): Validation target.
        - features (list): List of feature names.
        - target (str): The name of the target variable (default 'LAI').
        - importance (bool): Whether to compute and display feature importance.

        Returns:
        - model (RandomForestRegressor): Trained model.
        - feature_importance_df (DataFrame): DataFrame with feature importance (if importance=True).
        """
        # Create the output directory
        os.makedirs(self.name, exist_ok=True)
        os.chdir(self.name)

        # Initialize and train the RandomForest model
        model = RandomForestRegressor(
            n_estimators=self.params['n_estimators'],
            min_samples_split=self.params['min_samples_split'],
            criterion=self.params['criterion'],
            max_features=self.params['max_features'],
            min_samples_leaf=self.params['min_samples_leaf'],
            random_state=self.params['random_state']
        )
        model.fit(X_train, y_train)

        # Make predictions and compute error metrics
        predictions_test = model.predict(X_val)
        r, r2, mse, rmse, rrmse, mae, nash = error_metrics(y_val, predictions_test)

        # Print validation metrics
        val_metrics = {
            'r': r,
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'rrmse': rrmse,
            'mae': mae,
            'nash': nash
        }
        print('Validation metrics:', val_metrics)

        # Save the trained model
        joblib.dump(model, f'{self.name}_model')

        # Compute and save feature importance (if requested)
        if importance:
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            # Plot and save feature importance
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='royalblue')
            plt.xlabel('Importance')
            plt.title(f'{target} - Feature Importance')
            plt.gca().invert_yaxis()
            plt.savefig(f'{self.name}_importance.png')
            plt.show()

            # Save feature importance to Excel
            feature_importance_df.to_excel(f'{self.name}_feature_importance.xlsx', index=False)
        else:
            feature_importance_df = None

        # Prepare validation metrics and hyperparameters for saving
        metrics_df = pd.DataFrame({
            'Metric': ['r', 'r^2', 'MSE', 'RMSE', 'RRMSE', 'MAE', 'Nash-Sutcliffe'],
            'Value': [r, r2, mse, rmse, rrmse, mae, nash]
        })

        hyperparams_df = pd.DataFrame({
            'Parameter': ['n_estimators', 'criterion', 'max_features', 'min_samples_split', 'min_samples_leaf', 'random_state'],
            'Value': [self.params['n_estimators'], self.params['criterion'], self.params['max_features'], self.params['min_samples_split'], self.params['min_samples_leaf'], self.params['random_state']]
        })

        # Save metrics, hyperparameters, and feature importance to Excel
        with pd.ExcelWriter(f"{self.name}_validation_metrics.xlsx") as writer:
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            hyperparams_df.to_excel(writer, sheet_name='Hyperparameters', index=False)
            if importance:
                feature_importance_df.to_excel(writer, sheet_name='Feature Importance', index=False)

        return model


    def test(self, X_test, y_test, model):
        """Test the trained model."""

        os.makedirs(f'test', exist_ok=True)
        os.chdir(f'test')

        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
        elif isinstance(X_test, pd.DataFrame):
            pass
        else:
            raise TypeError("X_test must be an ndarray or DataFrame.")

        predictions_test = model.predict(X_test)

        r, r2, mse, rmse, rrmse, mae, nash = error_metrics(y_test, predictions_test)
        print(f"Testing RF RMSE: {rmse}")

        textstr = '\n'.join((
            r'$R^2=%.2f$' % (r2,),
            r'$RMSE=%.2f$' % (rmse,),
            r'$RRMSE=%.2f\%%$' % (rrmse,),
            r'$MAE=%.2f$' % (mae,),
            r'$NASH=%.2f$' % (nash,)
        ))

        plt.figure(figsize=(8, 6))
        plt.scatter(predictions_test, y_test)
        plt.plot([predictions_test.min(), predictions_test.max()], [predictions_test.min(), predictions_test.max()], '--', color='gray')

        m, b = np.polyfit(predictions_test.flatten(), y_test, 1)
        plt.plot(predictions_test, m * predictions_test + b, color='grey', lw=2)

        plt.xlabel('Predicted Value RF', fontsize=16)
        plt.ylabel('Measured Value', fontsize=16)
        plt.legend()
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=16, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        plt.tight_layout()
        plt.savefig(f'{self.name}_inference_rf.png')
        plt.show()



    def predict(self, model, min_max_values):

        os.chdir(self.original_dir)
        dataset = pd.read_excel(self.predict_config['dataset'])
        normalize = self.predict_config['normalize']
        features = self.predict_config['features']
        target = self.predict_config['target']

        dataset_inputs = dataset[features]


        if normalize and min_max_values is not None:
            def normalize_difference(df, columns, min_values=None, max_values=None):
                normalized_df = df.copy()
                for col in columns:
                    xmin = min_values[col] if min_values is not None else df[col].min()
                    xmax = max_values[col] if max_values is not None else df[col].max()
                    normalized_df[col] = (df[col] - xmin) / (xmax - xmin)
                return normalized_df

            min_values = min_max_values.set_index("variable")["min"].to_dict()
            max_values = min_max_values.set_index("variable")["max"].to_dict()
            inputs_array = normalize_difference(dataset_inputs, features, min_values=min_values, max_values=max_values)
        else:
            inputs_array = dataset_inputs.values

        start_time = time.time()
        predictions = model.predict(inputs_array)
        elapsed_time = time.time() - start_time
        print(f"Time taken for prediction: {elapsed_time} seconds")

        dataset_inputs[target] = predictions
        dataset[target] = predictions

        path_save = rf'{self.original_dir}/{self.name}'
        dataset.to_excel(f'{path_save}/outputs_{self.name}.xlsx', index=False)

        return dataset
