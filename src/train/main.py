import argparse
import logging

import torch

from src.client.camera import run_camera
from src.common.const import CONFIG
from src.common.utils import get_path, setup_logging
from src.train.model import get_model_from_path, load_model_from_path
from src.train.plots import plot_stats
from src.train.train import evaluate, load_images_from_folder, train
from src.train.validation import validate_to_csv

setup_logging("logs/train.log")
_log = logging.getLogger(__name__)


def main() -> None:
    """Train the model or classify images and output scores to a CSV file."""
    model_path = get_path(args.model_path)
    model = get_model_from_path(model_path)

    if not args.is_validate:
        # Load images
        dataset_path = get_path(args.dataset_path)
        _log.info("Loading images from %s", dataset_path)
        train_loader, test_loader = load_images_from_folder(dataset_path)
        _log.info("Number of images: %d", len(train_loader))

        # Train
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        _log.info("Current training device: %s", device)

        training_stats = train(
            model, train_loader, test_loader, epochs=CONFIG.epochs, use_gpu=torch.cuda.is_available()
        )
        loss, accuracy = evaluate(model, test_loader, use_gpu=torch.cuda.is_available())
        _log.info("Validation loss: %f, accuracy: %f", loss, accuracy)

        if device == "cuda":
            # Move the model back to the CPU
            model = model.to("cpu")
        # Save the model
        _log.info("Saving model to %s", model_path)
        torch.save(model.state_dict(), model_path)
        # Plot the training statistics
        plot_stats(training_stats, model_path, accuracy)
        return

    # Validate
    load_model_from_path(model_path, model)
    model.eval()
    if args.dataset_path is None:
        # Run the camera if no dataset is provided
        run_camera(model)
    else:
        # If the dataset is provided, validate the model on the dataset
        dataset_path = get_path(args.dataset_path)
        validate_to_csv(dataset_path, model)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train the model or classify images and output scores to a CSV file.")
    parser.add_argument(
        "--validate",
        dest="is_validate",
        action="store_true",
        help="Perform validation on the given model using the specified image folder. "
        "If no folder is provided, webcam input will be used. If validate is not set, the model will be trained.",
    )
    parser.add_argument(
        "--model",
        dest="model_path",
        type=str,
        required=True,
        help="Specify path to save or load the trained model. Include the model name in the filename.",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_path",
        type=str,
        help="Path to the folder containing images for training or validation.",
    )
    args = parser.parse_args()
    main()
