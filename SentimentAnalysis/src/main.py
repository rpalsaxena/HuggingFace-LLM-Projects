from data.dataset import load_imdb_dataset
from models.sentiment_model import get_model_and_tokenizer, train_model
from utils.helpers import preprocess_data
# from dotenv import load_dotenv
# import os
# load_dotenv()

def main():
    # Load dataset
    dataset = load_imdb_dataset()

    # Get model and tokenizer
    model, tokenizer = get_model_and_tokenizer()

    # Preprocess data
    tokenized_datasets = preprocess_data(tokenizer, dataset)

    # Split dataset
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    # Train model
    trainer = train_model(model, tokenizer, train_dataset, eval_dataset)

    # Evaluate model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

if __name__ == "__main__":
    main()