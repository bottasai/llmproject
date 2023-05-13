import csv
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForSequenceClassification

def summarize_reviews_with_sentiment_analysis(csv_file, output_file):
    # Load the T5 tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    # Load the BERT tokenizer and model for sentiment analysis
    sentiment_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sentiment_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    reviews = []

    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            review = row['Review'].strip()
            reviews.append(review)

    rows = []
    for review in reviews:
        # Perform sentiment analysis on the review
        inputs = sentiment_tokenizer(review, truncation=True, padding=True, return_tensors='pt')
        outputs = sentiment_model(**inputs)
        logits = outputs.logits
        predicted_labels = logits.argmax(dim=1)
        sentiment = 'Positive' if predicted_labels == 1 else 'Negative'

        # Prepare the input data
        input_text = "summarize: " + review
        inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)

        # Generate the summary
        summary_ids = model.generate(inputs, num_beams=4, max_length=150, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        rows.append([review, sentiment, summary])

    # Write output to a new CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Review', 'Sentiment', 'Summary'])  # Write header row
        writer.writerows(rows)

    print(f"Output written to {output_file} successfully.")

# Example usage
csv_file = 'trustpilot_reviews.csv'  # Replace with your CSV file path
output_file = 'output.csv'  # Replace with the desired output file path

summarize_reviews_with_sentiment_analysis(csv_file, output_file)
