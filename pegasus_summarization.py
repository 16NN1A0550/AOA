from transformers import PegasusForConditionalGeneration, AutoTokenizer
import torch

def pegasus_summarizer(raw_text):
    model_name = 'google/pegasus-cnn_dailymail'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device="cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    batch = tokenizer(raw_text, truncation=True, padding='longest', return_tensors="pt").to(device)
    # Specify the percentage of the input length you want for the summary
    percentage_max = 15  # For example, 30% of the input length
    percentage_min = 10
    # Calculate the max_length for the summary as a percentage of the input length
    max_length = int(len(raw_text) * (percentage_max / 100))
    min_length = int(len(raw_text) * (percentage_min / 100))
    translated = model.generate(**batch,max_length=max_length, min_length=min_length, num_beams=6,
        length_penalty=1.2,
        temperature=0.7,
        early_stopping=True,
        do_sample=False)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    tgt_text="".join(tgt_text)
    tgt_text=tgt_text.replace("<n>","").strip()
    return tgt_text