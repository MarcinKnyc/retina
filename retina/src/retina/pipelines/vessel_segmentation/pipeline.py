from kedro.pipeline import Pipeline, node

from .feature_extraction import create_dataset, preprocess_images
from .loading import load_data
from .models_and_predictions import predict_model, train_adaboost, train_knn, train_logitboost, undersampling
from .visualisation import plot_images, plot_results

from .prepare import prepare_stare, prepare_drive


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=prepare_stare,
                inputs=["params:stare_images_url",
                        "params:stare_labels_url",
                        "params:stare_path"],
                outputs="sequence",  # TODO: use this to force execution before other nodes
                name="STARE"
            ),
            node(func=prepare_drive,
                 inputs=["params:datapath",
                         "params:drive_path"],
                 outputs="sequence2",  # TODO: use this to force execution before other nodes
                 name="DRIVE"
                 ),
            node(
                func=load_data,
                # inputs=["params:kaggle_rename_path_to",
                inputs=["params:stare_path",
                        "params:debug"],
                outputs=["train_raw_photos",
                         "train_masks",
                         "test_raw_photos",
                         "test_masks"],
                name="load_data_node",
            ),
            node(
                func=preprocess_images,
                inputs=["train_raw_photos",
                        "test_raw_photos"],
                outputs=["train_photos",
                         "test_photos"],
                name="preprocess_images_node",
            ),
            node(
                func=plot_images,
                inputs=["train_raw_photos",
                        "train_masks",
                        "params:image_plot_title",
                        "params:output_path",
                        "params:image_plot_filename"],
                outputs=None,
                name="plot_images_node",
            ),
            node(
                func=create_dataset,
                inputs=["train_photos",
                        "train_masks"],
                outputs=["train_features",
                         "train_labels"],
                name="create_dataset_node",
            ),
            node(
                func=undersampling,
                inputs=["train_features", "train_labels"],
                outputs=["train_features_under",
                         "train_labels_under"],
                name="undersampling_node",
            ),
            node(
                func=train_logitboost,
                inputs=["train_features_under",
                        "train_labels_under",
                        "params:output_path"],
                outputs="knn_classifier",
                name="train_model_node",
            ),
            node(
                func=predict_model,
                inputs=["knn_classifier",
                        "test_photos",
                        "test_masks"],
                outputs="y_pred_images",
                name="predict_model_node",
            ),
            node(
                func=plot_results,
                inputs=["test_photos",
                        "test_masks",
                        "y_pred_images",
                        "params:result_plot_title",
                        "params:output_path",
                        "params:result_plot_filename"],
                outputs=None,
                name="plot_results_node",
            ),
        ]
    )
