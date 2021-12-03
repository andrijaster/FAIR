from process import _process
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hyperparameters",
        type=list,
        default=[2, 2, 2, 2, 1.5, 1.5, 1.5, 1.5],
        help="[num_layers_z, num_layers_y,num_layers_w, num_layers_A, step_z, step_y, step_A, step_w]",
    )
    parser.add_argument("--epochs", type=int, default=1, help="no. of epochs")

    parser.add_argument(
        "--threshold", type=float, default=0.5, help="threshold for output evaluation"
    )

    parser.add_argument(
        "--alpha",
        type=list,
        default=[0, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
        help="range of alpha hyperparameter",
    )

    parser.add_argument(
        "--saver_dir_models",
        type=str,
        default="data/Trained_models",
        help="directory where models are going to be saved",
    )

    parser.add_argument(
        "--saver_dir_results",
        type=str,
        default="data/Results",
        help="directory where results are going to be saved",
    )

    parser.add_argument("--name", type=str, default="Medical_exp", help="model name")

    args = parser.parse_args()

    (
        num_layers_z,
        num_layers_y,
        num_layers_w,
        num_layers_A,
        step_z,
        step_y,
        step_A,
        step_w,
    ) = args.hyperparameters

    _process(
        num_layers_z=num_layers_z,
        num_layers_y=num_layers_y,
        num_layers_w=num_layers_w,
        num_layers_A=num_layers_A,
        step_z=step_z,
        step_y=step_y,
        step_A=step_A,
        step_w=step_w,
        epochs=args.epochs,
        threshold=args.threshold,
        alpha=args.alpha,
        saver_dir_models=args.saver_dir_models,
        saver_dir_results=args.saver_dir_results,
        name=args.name,
    )
