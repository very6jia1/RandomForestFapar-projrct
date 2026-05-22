from processing.splits_processing import DatasetProcessor
from processing.RF_processing import RFProcessor
from config.splits_params import splits_params
from config.RF_params import RF_params, RF_predict


def main():
    # SPLITS
    processorSplits = DatasetProcessor(splits_params)
    processorSplits.normalize_splits()
    combined_X, combined_y, min_max_values = processorSplits.save_splits()
    X_train = combined_X['train']
    y_train = combined_y['train']
    X_val = combined_X['val']
    y_val = combined_y['val']
    X_test = combined_X['test']
    y_test = combined_y['test']

    # RF
    processorRF = RFProcessor('outputs/trial1_RF', RF_params, RF_predict)

    feature_list = [str(f) for f in splits_params['features']]

    modelRF = processorRF.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        importance=True,
        features=feature_list,
        target=str(splits_params['target'])
    )

    processorRF.test(X_test=X_test, y_test=y_test, model=modelRF)
    processorRF.predict(model=modelRF, min_max_values=min_max_values)


if __name__ == '__main__':
    main()
