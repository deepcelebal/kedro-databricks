from kedro.pipeline import Pipeline, node
from .nodes import oversample_data, split_data, train_model, evaluate_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=oversample_data,
                inputs=['preprocessed_stroke_data', 'parameters'],
                outputs=['X_res', 'y_res'],
                name = 'oversample_data_node'
            ),
            node(
            func= split_data,
            inputs=["X_res", 'y_res', 'parameters'],
            outputs=['X_train', 'X_test', 'y_train', 'y_test'],
            name="split_data_node",
        ),
        node(
            func= train_model,
            inputs=["X_train", "y_train", 'parameters'],
            outputs="classifier",
            name= "train_model_node",
        ),
        node(
            func=evaluate_model,
            inputs=['classifier', 'X_test', 'y_test'],
            outputs = None,
            name= 'evaluate_model_node',
        ),
        ]
    )