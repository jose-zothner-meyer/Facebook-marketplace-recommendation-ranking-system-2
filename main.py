from pipeline import Pipeline  # Import the Pipeline class from pipeline.py to handle various stages of the project.

def main():
    """
    Main entry point for the application.

    This function initializes and runs the Pipeline with the options to process data, process images,
    evaluate the model, and train the model.
    """
    # Initialize a Pipeline instance with all processing flags set to True.
    # process_data: Whether to perform data processing.
    # process_imgs: Whether to process images (e.g., cleaning, resizing).
    # eval_model: Whether to evaluate the existing model.
    # train_model: Whether to train the model.
    Pipeline(process_data=True, process_imgs=True, eval_model=True, train_model=True)

# This conditional ensures that main() is called only when this script is executed directly,
# and not when imported as a module in another script.
if __name__ == "__main__":
    main()