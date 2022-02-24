from kedro.pipeline import Pipeline, node
from .nodes import preprocess_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
            func= preprocess_data,
            inputs="stroke_raw_data",
            outputs="preprocessed_stroke_data",
            name="preprocess_stroke_node"
        )
        ]
    )