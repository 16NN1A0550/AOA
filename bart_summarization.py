from transformers import pipeline
from transformers import BartForConditionalGeneration, BartTokenizer

model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def bart_summarizer(raw_text):
    inputs = tokenizer.encode(raw_text, return_tensors="pt", max_length=1024, truncation=True)
    # Specify the percentage of the input length you want for the summary
    percentage_max = 15  # For example, 30% of the input length
    percentage_min = 10
    # Calculate the max_length for the summary as a percentage of the input length
    max_length = int(len(raw_text) * (percentage_max / 100))
    min_length = int(len(raw_text) * (percentage_min / 100))
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, num_beams=6,
        length_penalty=1.2,
        temperature=0.7,
        early_stopping=True,
        do_sample=False)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    formatted_summary = "".join(summary)
    return formatted_summary

