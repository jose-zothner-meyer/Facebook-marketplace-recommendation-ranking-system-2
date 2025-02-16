from pipeline import Pipeline

def main():
    Pipeline(process_data=True, process_imgs=True, eval_model=True, train_model=True)


if __name__ == "__main__":
    main()

