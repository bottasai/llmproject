import csv
from transformers import T5Tokenizer, T5ForConditionalGeneration

def summarize_reviews_from_csv(csv_file):
    # Load the T5 tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    reviews = []

    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            review = row['Review'].strip()
            reviews.append(review)

    summaries = []
    for review in reviews:
        # Prepare the input data
        input_text = "summarize: " + review
        inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)

        # Generate the summary
        summary_ids = model.generate(inputs, num_beams=4, max_length=50, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=False)
        summaries.append(summary)

    return summaries

# Example usage
csv_file = 'trustpilot_reviews.csv'  # Replace with your CSV file path

summaries = summarize_reviews_from_csv(csv_file)

for i, summary in enumerate(summaries):
    print(f"Review {i+1} Summary:", summary)
