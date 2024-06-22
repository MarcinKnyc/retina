from kedro.pipeline import Pipeline, node

from .feature_extraction import create_dataset, extract_features
from .models_and_predictions import predict_model, train_adaboost, train_knn, train_logitboost, undersampling, apply_threshold
from .visualisation import plot_images, plot_results, plot_features

from .prepare import prepare_stare, prepare_drive, prepare_chasedb1
from kedro.pipeline.modular_pipeline import pipeline

def create_pipeline(**kwargs):

    classifiers = (
        ("logitboost", train_logitboost),
        ("knn", train_knn),
        ("adaboost", train_adaboost),
        )
    datasets = (
            "drive",
            "stare",
            "chasedb1"
        )

    image_prepare_pipeline = Pipeline(
        [
            node(
                func=plot_images,
                inputs=[f"train_photos",
                        f"train_masks",
                        "params:image_plot_title",
                        "params:output_path",
                        "params:image_plot_filename"],
                outputs=None,
                name=f"plot_images",
            ),
            node( 
                func=extract_features,
                inputs="train_photos",
                outputs="train_feature_photos",
                name="extract_training_features"
            ),
            node(
                func = plot_features,
                inputs = ["train_feature_photos", 
                    "params:output_path"],
                outputs = None,
                name="plot_features"
            ),
            node( 
                func=extract_features,
                inputs="test_photos",
                outputs="test_feature_photos",
                name="extract_testing_features"
            ),
            node(
                func=create_dataset,
                inputs=["train_photos",
                        f"train_feature_photos",
                        f"train_masks"],
                outputs=[f"train_features",
                        f"train_labels"],
                name=f"dataset",
            ),
            node(
                func=undersampling,
                inputs=[f"train_features",
                        f"train_labels",
                        "params:sample_size"],
                outputs=[f"train_features_under",
                        f"train_labels_under"],
                name=f"undersampling",
            ),
        ]

    )

    validation_pipeline_template = Pipeline([ node(
                func=predict_model,
                inputs=[f"classifier",
                        f"test_feature_photos",
                        "test_photos"
                        ],
                outputs=f"pred_images",
                name=f"predict_model",
            ),
            node(
                func=plot_results,
                inputs=[f"test_photos",
                        f"test_masks",
                        "params:threshold",
                        f"pred_images",
                        f"params:result_plot_title",
                        "params:output_path",
                        f"params:result_plot_filename"],
                outputs=None,
                name=f"plot_results",
            ),]
    )

    stare_preparation_pipeline = pipeline(
        [
            node(
                func=prepare_stare,
                inputs=["params:stare_path",
                        "params:stare_images_url",
                        "params:stare_labels_url",
                        "params:debug"],
                outputs=[f"train_photos",
                        f"train_masks",
                        f"test_photos",
                        f"test_masks"],
                name=f"STARE"
            ),
            image_prepare_pipeline,
        ],
        parameters={ "params:stare_path":"params:stare_path",
            "params:stare_images_url":"params:stare_images_url",
            "params:stare_labels_url":"params:stare_labels_url",
            "params:debug":"params:debug",
            "params:output_path":"params:stare_output_path",
            "params:image_plot_title":"params:stare_image_plot_title",
            "params:image_plot_filename":"params:stare_image_plot_filename",
            "params:sample_size":"params:stare_sample_size"
            },
        namespace = "stare"
    )

    chasedb1_preparation_pipeline = pipeline(
        [
            node(
                func=prepare_chasedb1,
                inputs=["params:chasedb1_path",
                        "params:chasedb1_url",
                        "params:debug"],
                outputs=[f"train_photos",
                        f"train_masks",
                        f"test_photos",
                        f"test_masks"],
                name=f"CHASEDB1"
            ),
            image_prepare_pipeline,
        ],
        parameters={ "params:chasedb1_path":"params:chasedb1_path",
            "params:chasedb1_url":"params:chasedb1_url",
            "params:debug":"params:debug",
            "params:output_path":"params:chasedb1_output_path",
            "params:image_plot_title":"params:chasedb1_image_plot_title",
            "params:image_plot_filename":"params:chasedb1_image_plot_filename",
            "params:sample_size":"params:chasedb1_sample_size"
            },
        namespace = "chasedb1"
    )

    drive_preparation_pipeline = pipeline(
        [
            node(
                func=prepare_drive,
                inputs=["params:drive_path",
                        "params:debug"],
                outputs=[f"train_photos",
                        f"train_masks",
                        f"test_photos",
                        f"test_masks"],
                name=f"DRIVE"
            ),
            image_prepare_pipeline,
        ],
        parameters={ 
            "params:drive_path":"params:drive_path",
            "params:debug":"params:debug",
            "params:output_path":"params:drive_output_path",
            "params:image_plot_title":"params:drive_image_plot_title",
            "params:image_plot_filename":"params:drive_image_plot_filename",
            "params:sample_size":"params:drive_sample_size"
            },
        namespace = "drive"
    )

    classifier_training_pipelines = []

    for classifier_name, classifier in classifiers:
        for dataset in (
            "stare",
            "drive",
            "chasedb1"
        ):
            classifier_pipeline = pipeline(
                [
                    node(
                        func=classifier,
                        inputs=["train_features_under",
                                "train_labels_under",],
                        outputs="classifier",
                        name=f"train_{classifier_name}_on_{dataset}",
                    ),
                ],
                inputs={"train_features_under":f"{dataset}.train_features_under",
                "train_labels_under":f"{dataset}.train_labels_under"
                },
                outputs = {"classifier":f"{classifier_name}_{dataset}"}
            )

            classifier_training_pipelines.append( classifier_pipeline )

    dataset_pipeline = stare_preparation_pipeline + drive_preparation_pipeline + chasedb1_preparation_pipeline

    classifier_pipeline = pipeline(classifier_training_pipelines)

    validation_pipelines=[]

    for classifier_name, classifier in classifiers:
        for training_dataset in datasets:
            for testing_dataset in datasets:
                if training_dataset == testing_dataset:
                    result_plot_title = f"params:{training_dataset}_result_{classifier_name}_plot_title"
                    result_plot_filename = f"params:{training_dataset}_result_{classifier_name}_plot_filename"
                else:
                    result_plot_title = f"params:{training_dataset}_{testing_dataset}_result_{classifier_name}_plot_title"
                    result_plot_filename = f"params:{training_dataset}_{testing_dataset}_result_{classifier_name}_plot_filename"
                validation_pipeline = pipeline(
                    validation_pipeline_template,
                    inputs={
                        "classifier":f"{classifier_name}_{training_dataset}",
                        "test_feature_photos":f"{testing_dataset}.test_feature_photos",
                        "test_photos":f"{testing_dataset}.test_photos",
                        "test_masks":f"{testing_dataset}.test_masks"
                    },
                    parameters={
                        f"params:result_plot_title":
                        result_plot_title,
                        "params:output_path":f"params:{testing_dataset}_output_path",
                        f"params:result_plot_filename":result_plot_filename,
                        "params:threshold":"params:threshold"
                    },
                    namespace=f"{training_dataset}_{testing_dataset}_{classifier_name}"
                )
                validation_pipelines.append(validation_pipeline)

    validation_pipeline = pipeline(validation_pipelines)

    output_pipeline = dataset_pipeline + classifier_pipeline + validation_pipeline
    return output_pipeline
