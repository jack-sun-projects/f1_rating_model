from loguru import logger

from f1_rating_model.src.model.model_service import ModelService
from f1_rating_model.src.model.pipeline.data_preparation import prepare_data
from f1_rating_model.src.model.pipeline.train_model import _split_X_y


def main():
    logger.info('Running application')

    results_full, results_predictions = prepare_data()

    X_col_drop = [
        'year',
        'round',
        'date',
        'race_name',
        'num_finishing_drivers',
        'dob',
        'status',
        'positionOrder',
        'position_percentile',
    ]
    y_col = 'position_percentile'
    X, _ = _split_X_y(results_full, X_col_drop, y_col)

    ml_svc = ModelService()
    ml_svc.load_model('f1_v1')
    rankings = ml_svc.generate_rankings(X,
                                        results_predictions,
                                        ranking_type="annual",
                                        year=2024)
    print(rankings)


if __name__ == '__main__':
    main()
