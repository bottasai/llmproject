import csv
import datetime
from transformers import T5Tokenizer, T5ForConditionalGeneration
from textblob import TextBlob

def summarize_reviews_with_sentiment_analysis(csv_file):
    # Load the T5 tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    reviews = []

    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            review = row['Review'].strip()
            reviews.append(review)

    rows = []
    for review in reviews:
        # Perform sentiment analysis on the review
        sentiment = TextBlob(review).sentiment.polarity
        #sentiment_label = 'Positive' if sentiment >= 0 else 'Negative'

        # Prepare the input data
        input_text = "summarize: " + review
        inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)

        # Generate the summary
        summary_ids = model.generate(inputs, num_beams=4, max_length=150, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        rows.append([review, sentiment*10, summary])

    # Get the current date and timestamp
    current_datetime = datetime.datetime.now()
    timestamp = current_datetime.strftime("%Y%m%d_%H%M%S")

    # Construct the output file name with the date and timestamp
    output_file_with_timestamp = f"output_{timestamp}.csv"

    # Write output to a new CSV file with the timestamp included
    with open(output_file_with_timestamp, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Review', 'Sentiment', 'Summary'])  # Write header row
        writer.writerows(rows)

    print(f"Output written to {output_file_with_timestamp} successfully.")

# Example usage
csv_file = 'trustpilot_reviews_1.csv'  # Replace with your CSV file path

summarize_reviews_with_sentiment_analysis(csv_file)
