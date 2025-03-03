import argparse
import os
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, AutoConfig
import transformers
from tqdm import tqdm
from scipy.special import softmax
import numpy as np


# Load Model and Tokenizer
transformers.logging.set_verbosity_error()
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, config=config)

def process_sentiment(text):
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    s = {}
    for i in range(scores.shape[0]):
        l = config.id2label[ranking[i]]
        s[l]= float(scores[ranking[i]])
    return s

def analyze_sentiment(text):
    """Analyze sentiment of a given text."""
    result = process_sentiment(text)
    return result

def process_csv(input_csv, output_csv):
    """Process a CSV file and add sentiment analysis results."""
    df = pd.read_csv(input_csv)

    if df.shape[1] != 1:
        raise ValueError("CSV file should contain only one column of text data.")

    column_name = df.columns[0]

    sentiments = []
    positive_scores = []
    neutral_scores = []
    negative_scores = []

    for text in tqdm(df[column_name], desc="Processing"):
        try:
            result = process_sentiment(text)
            overall = list(result.keys())[0]
            sentiments.append(overall)
            positive_scores.append(result["positive"])
            neutral_scores.append(result["neutral"])
            negative_scores.append(result["negative"])
        except Exception as e:
            print(f"Error processing text: {text}, Error: {e}")
            sentiments.append("ERROR")
            positive_scores.append(0)
            neutral_scores.append(0)
            negative_scores.append(0)

    df["Sentiment"] = sentiments
    df["Positive_Score"] = positive_scores
    df["Neutral_Score"] = neutral_scores
    df["Negative_Score"] = negative_scores

    df.to_csv(output_csv, index=False)
    print(f"Processed file saved as {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis CLI Tool using Twitter RoBERTa.")
    parser.add_argument("--csv", type=str, help="Path to input CSV file with a single column of text.")
    parser.add_argument("--text", type=str, help="Single text message for sentiment analysis.")
    parser.add_argument("--output", type=str, default=None, help="Path to save output CSV file (if processing a CSV).")

    args = parser.parse_args()

    if args.csv:
        output_path = args.output if args.output else os.path.join(os.getcwd(), "output.csv")
        process_csv(args.csv, output_path)
    elif args.text:
        result:dict = analyze_sentiment(args.text)
        for sentiment, score in result.items():
            print(f"Sentiment: {sentiment}, Score: {score}")
        overall_sentiment = list(result.keys())[0]
        print(f"Overall sentiment: {overall_sentiment}")
    else:
        print("Please provide either a CSV file (--csv) or a text string (--text).")

if __name__ == "__main__":
    main()
