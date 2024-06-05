from kedro.pipeline import Pipeline, node

from .feature_extraction import create_dataset
from .models_and_predictions import predict_model, train_adaboost, train_knn, train_logitboost, undersampling
from .visualisation import plot_images, plot_results

from .prepare import prepare_stare, prepare_drive, prepare_chasedb1


def create_pipeline(**kwargs):

    partial_pipelines = []

    for classifier_name, classifier in (
        ("logitboost", train_logitboost),
        #("knn", train_knn),
        #("adaboost", train_adaboost)
    ):

        stare_pipeline = Pipeline(
            [
                node(
                    func=prepare_stare,
                    inputs=["params:stare_path",
                            "params:stare_images_url",
                            "params:stare_labels_url",
                            "params:debug"],
                    outputs=[f"{classifier_name}_STARE_train_photos",
                            f"{classifier_name}_STARE_train_masks",
                            f"{classifier_name}_STARE_test_photos",
                            f"{classifier_name}_STARE_test_masks"],
                    name=f"{classifier_name}_STARE"
                ),
                node(
                    func=plot_images,
                    inputs=[f"{classifier_name}_STARE_train_photos",
                            f"{classifier_name}_STARE_train_masks",
                            "params:stare_image_plot_title",
                            "params:stare_output_path",
                            "params:stare_image_plot_filename"],
                    outputs=None,
                    name=f"{classifier_name}_plot_STARE",
                ),
                node(
                    func=create_dataset,
                    inputs=[f"{classifier_name}_STARE_train_photos",
                            f"{classifier_name}_STARE_train_masks"],
                    outputs=[f"{classifier_name}_STARE_train_features",
                            f"{classifier_name}_STARE_train_labels"],
                    name=f"{classifier_name}_STARE_dataset",
                ),
                node(
                    func=undersampling,
                    inputs=[f"{classifier_name}_STARE_train_features",
                            f"{classifier_name}_STARE_train_labels"],
                    outputs=[f"{classifier_name}_STARE_train_features_under",
                            f"{classifier_name}_STARE_train_labels_under"],
                    name=f"{classifier_name}_STARE_undersampling",
                ),
                node(
                    func=classifier,
                    inputs=[f"{classifier_name}_STARE_train_features_under",
                            f"{classifier_name}_STARE_train_labels_under",],
                    outputs=f"{classifier_name}_STARE_knn_classifier",
                    name=f"{classifier_name}_STARE_train",
                ),
                node(
                    func=predict_model,
                    inputs=[f"{classifier_name}_STARE_knn_classifier",
                            f"{classifier_name}_STARE_test_photos",
                            f"{classifier_name}_STARE_test_masks"],
                    outputs=f"{classifier_name}_STARE_pred_images",
                    name=f"{classifier_name}_STARE_predict_model",
                ),
                node(
                    func=plot_results,
                    inputs=[f"{classifier_name}_STARE_test_photos",
                            f"{classifier_name}_STARE_test_masks",
                            f"{classifier_name}_STARE_pred_images",
                            f"params:stare_result_{classifier_name}_plot_title",
                            "params:stare_output_path",
                            f"params:stare_result_{classifier_name}_plot_filename"],
                    outputs=None,
                    name=f"{classifier_name}_STARE_plot_results",
                ),
            ]
        )

        drive_pipeline = Pipeline(
            [
                node(
                    func=prepare_drive,
                    inputs=["params:drive_path",
                            "params:debug"],
                    outputs=[f"{classifier_name}_DRIVE_train_photos",
                            f"{classifier_name}_DRIVE_train_masks",
                            f"{classifier_name}_DRIVE_test_photos",
                            f"{classifier_name}_DRIVE_test_masks"],
                    name=f"{classifier_name}_DRIVE"
                ),
                node(
                    func=plot_images,
                    inputs=[f"{classifier_name}_DRIVE_train_photos",
                            f"{classifier_name}_DRIVE_train_masks",
                            "params:drive_image_plot_title",
                            "params:drive_output_path",
                            "params:drive_image_plot_filename"],
                    outputs=None,
                    name=f"{classifier_name}_plot_DRIVE",
                ),
                node(
                    func=create_dataset,
                    inputs=[f"{classifier_name}_DRIVE_train_photos",
                            f"{classifier_name}_DRIVE_train_masks"],
                    outputs=[f"{classifier_name}_DRIVE_train_features",
                            f"{classifier_name}_DRIVE_train_labels"],
                    name=f"{classifier_name}_DRIVE_dataset",
                ),
                node(
                    func=undersampling,
                    inputs=[f"{classifier_name}_DRIVE_train_features",
                            f"{classifier_name}_DRIVE_train_labels"],
                    outputs=[f"{classifier_name}_DRIVE_train_features_under",
                            f"{classifier_name}_DRIVE_train_labels_under"],
                    name=f"{classifier_name}_DRIVE_undersampling",
                ),
                node(
                    func=classifier,
                    inputs=[f"{classifier_name}_DRIVE_train_features_under",
                            f"{classifier_name}_DRIVE_train_labels_under",],
                    outputs=f"{classifier_name}_DRIVE_knn_classifier",
                    name=f"{classifier_name}_DRIVE_train",
                ),
                node(
                    func=predict_model,
                    inputs=[f"{classifier_name}_DRIVE_knn_classifier",
                            f"{classifier_name}_DRIVE_test_photos",
                            f"{classifier_name}_DRIVE_test_masks"],
                    outputs=f"{classifier_name}_DRIVE_pred_images",
                    name=f"{classifier_name}_DRIVE_predict_model",
                ),
                node(
                    func=plot_results,
                    inputs=[f"{classifier_name}_DRIVE_test_photos",
                            f"{classifier_name}_DRIVE_test_masks",
                            f"{classifier_name}_DRIVE_pred_images",
                            f"params:drive_result_{classifier_name}_plot_title",
                            "params:drive_output_path",
                            f"params:drive_result_{classifier_name}_plot_filename"],
                    outputs=None,
                    name=f"{classifier_name}_DRIVE_plot_results",
                ),
            ]
        )

        chasedb1_pipeline = Pipeline(
            [
                node(
                    func=prepare_chasedb1,
                    inputs=["params:chasedb1_path",
                            "params:chasedb1_url",
                            "params:debug"],
                    outputs=[f"{classifier_name}_CHASEDB1_train_photos",
                            f"{classifier_name}_CHASEDB1_train_masks",
                            f"{classifier_name}_CHASEDB1_test_photos",
                            f"{classifier_name}_CHASEDB1_test_masks"],
                    name=f"{classifier_name}_CHASEDB1"
                ),
                node(
                    func=plot_images,
                    inputs=[f"{classifier_name}_CHASEDB1_train_photos",
                            f"{classifier_name}_CHASEDB1_train_masks",
                            "params:chasedb1_image_plot_title",
                            "params:chasedb1_output_path",
                            "params:chasedb1_image_plot_filename"],
                    outputs=None,
                    name=f"{classifier_name}_plot_CHASEDB1",
                ),
                node(
                    func=create_dataset,
                    inputs=[f"{classifier_name}_CHASEDB1_train_photos",
                            f"{classifier_name}_CHASEDB1_train_masks"],
                    outputs=[f"{classifier_name}_CHASEDB1_train_features",
                            f"{classifier_name}_CHASEDB1_train_labels"],
                    name=f"{classifier_name}_CHASEDB1_dataset",
                ),
                node(
                    func=undersampling,
                    inputs=[f"{classifier_name}_CHASEDB1_train_features",
                            f"{classifier_name}_CHASEDB1_train_labels"],
                    outputs=[f"{classifier_name}_CHASEDB1_train_features_under",
                            f"{classifier_name}_CHASEDB1_train_labels_under"],
                    name=f"{classifier_name}_CHASEDB1_undersampling",
                ),
                node(
                    func=classifier,
                    inputs=[f"{classifier_name}_CHASEDB1_train_features_under",
                            f"{classifier_name}_CHASEDB1_train_labels_under",],
                    outputs=f"{classifier_name}_CHASEDB1_knn_classifier",
                    name=f"{classifier_name}_CHASEDB1_train",
                ),
                node(
                    func=predict_model,
                    inputs=[f"{classifier_name}_CHASEDB1_knn_classifier",
                            f"{classifier_name}_CHASEDB1_test_photos",
                            f"{classifier_name}_CHASEDB1_test_masks"],
                    outputs=f"{classifier_name}_CHASEDB1_pred_images",
                    name=f"{classifier_name}_CHASEDB1_predict_model",
                ),
                node(
                    func=plot_results,
                    inputs=[f"{classifier_name}_CHASEDB1_test_photos",
                            f"{classifier_name}_CHASEDB1_test_masks",
                            f"{classifier_name}_CHASEDB1_pred_images",
                            f"params:chasedb1_result_{classifier_name}_plot_title",
                            "params:chasedb1_output_path",
                            f"params:chasedb1_result_{classifier_name}_plot_filename"],
                    outputs=None,
                    name=f"{classifier_name}_CHASEDB1_plot_results",
                ),
            ]
        )

        dataset_names = ["stare", "drive", "chasedb1"]

        for dataset1 in dataset_names:
            for dataset2 in dataset_names:
                if dataset1 != dataset2:
                    partial_pipelines.append(
                        Pipeline(
                            [
                                node(
                                    func=predict_model,
                                    inputs=[f"{classifier_name}_{dataset1.upper()}_knn_classifier",
                                            f"{classifier_name}_{dataset2.upper()}_test_photos",
                                            f"{classifier_name}_{dataset2.upper()}_test_masks"],
                                    outputs=f"{classifier_name}_{dataset1.upper()}_{dataset2.upper()}_pred_images",
                                    name=f"{classifier_name}_{dataset1.upper()}_{dataset2.upper()}_predict_model",
                                ),
                                node(
                                    func=plot_results,
                                    inputs=[f"{classifier_name}_{dataset2.upper()}_test_photos",
                                            f"{classifier_name}_{dataset2.upper()}_test_masks",
                                            f"{classifier_name}_{dataset1.upper()}_{dataset2.upper()}_pred_images",
                                            f"params:{dataset1}_{dataset2}_result_{classifier_name}_plot_title",
                                            f"params:{dataset2}_output_path",
                                            f"params:{dataset1}_{dataset2}_result_{classifier_name}_plot_filename"],
                                    outputs=None,
                                    name=f"{classifier_name}_{dataset1.upper()}_{dataset2.upper()}_plot_results",
                                ),
                            ]
                        )
                    )

       

        partial_pipelines.append(stare_pipeline + drive_pipeline + chasedb1_pipeline)
    
    if len(partial_pipelines) == 0:
        return None
    pipeline = partial_pipelines[0]
    for partial_pipeline in partial_pipelines[1:]:
        pipeline += partial_pipeline
    return pipeline
