from __future__ import unicode_literals
from flask import Flask,render_template,url_for,request
from spacy_summarization import text_summarizer
#from gensim.summarization import summarize
from nltk_summarization import nltk_summarizer
from bart_summarization import bart_summarizer
from bert_summarization import bert_summarizer
from pegasus_summarization import pegasus_summarizer
import time
import spacy
from rouge_score import rouge_scorer
from bert_score import score
import nltk
nltk.download('wordnet')
from nltk.translate.meteor_score import meteor_score
nlp = spacy.load('en_core_web_sm')
app = Flask(__name__)

from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField

# Web Scraping Pkg
from bs4 import BeautifulSoup
# from urllib.request import urlopen
from urllib.request import urlopen

# Sumy Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Sumy 
def sumy_summary(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result


# Reading Time
def readingTime(mytext):
	total_words = len([ token.text for token in nlp(mytext)])
	estimatedTime = total_words/200.0
	return estimatedTime

# Fetch Text From Url
def get_text(url):
	page = urlopen(url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text

def calculate_compression_ratio(original_text, summary_text):
    # Calculate the length of the original and summarized text (in words)
    original_length = len(original_text.split())
    summary_length = len(summary_text.split())
    
    # Compression Ratio
    compression_ratio = summary_length / original_length
    return compression_ratio


def calculate_rouge(generated_summary, reference_summary):
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate ROUGE scores
    scores = scorer.score(reference_summary, generated_summary)
    return scores


def calculate_bert(generated_summary,reference_summary):
    
    # Calculate ROUGE scores
    scores = score(reference_summary, generated_summary,lang="en", rescale_with_baseline=True)
    return scores


app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['DEBUG'] = True

class MyForm(FlaskForm):
    file = FileField('Select a file')
    submit = SubmitField('Upload')


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/analyze',methods=['GET','POST'])
def analyze():
	start = time.time()
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		final_reading_time = readingTime(rawtext)
		final_summary = text_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary)
		end = time.time()
		final_time = end-start
	return render_template('index.html',ctext=rawtext,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)



@app.route('/analyze_file', methods=['GET', 'POST'])
def analyze_file():
	start = time.time()
	if request.method=='POST':
		file=request.files['file']
		file_contents=file.read().decode('utf-8')
		rawtext=file_contents
		final_summary = text_summarizer(rawtext)
		return render_template('index.html',ctext=rawtext,final_summary=final_summary)


@app.route('/analyze_url',methods=['GET','POST'])
def analyze_url():
	start = time.time()
	if request.method == 'POST':
		raw_url = request.form['raw_url']
		rawtext = get_text(raw_url)
		final_reading_time = readingTime(rawtext)
		final_summary = text_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary)
		end = time.time()
		final_time = end-start
	return render_template('index.html',ctext=rawtext,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)



@app.route('/compare_summary')
def compare_summary():
	return render_template('compare_summary.html')

@app.route('/comparer',methods=['GET','POST'])
def comparer():
	start = time.time()
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		final_reading_time = readingTime(rawtext)
		final_summary_spacy = text_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary_spacy)
		compress_ratio_spacy=calculate_compression_ratio(rawtext,final_summary_spacy)
		scores_spacy=calculate_rouge(final_summary_spacy,rawtext)
		rouge1_spacy=scores_spacy['rouge1']
		rouge2_spacy=scores_spacy['rouge2']
		rougel_spacy=scores_spacy['rougeL']
		meteor_spacy = meteor_score([final_summary_spacy.split()], rawtext.split())
		bert_score_spacy=calculate_bert([final_summary_spacy], [rawtext])
		bert_score_spacy_p=bert_score_spacy[0]
		bert_score_spacy_r=bert_score_spacy[1]
		bert_score_spacy_f=bert_score_spacy[2]
	
		# NLTK
		final_summary_nltk = nltk_summarizer(rawtext)
		summary_reading_time_nltk = readingTime(final_summary_nltk)
		compress_ratio_nltk=calculate_compression_ratio(rawtext,final_summary_nltk)
		scores_nltk=calculate_rouge(final_summary_nltk,rawtext)
		rouge1_nltk=scores_nltk['rouge1']
		rouge2_nltk=scores_nltk['rouge2']
		rougel_nltk=scores_nltk['rougeL']
		meteor_nltk = meteor_score([final_summary_nltk.split()], rawtext.split())
		bert_score_nltk=calculate_bert([final_summary_nltk], [rawtext])
		bert_score_nltk_p=bert_score_nltk[0]
		bert_score_nltk_r=bert_score_nltk[1]
		bert_score_nltk_f=bert_score_nltk[2]
	
		# Sumy
		final_summary_sumy = sumy_summary(rawtext)
		summary_reading_time_sumy = readingTime(final_summary_sumy) 
		compress_ratio_sumy=calculate_compression_ratio(rawtext,final_summary_sumy)
		scores_sumy=calculate_rouge(final_summary_sumy,rawtext)
		rouge1_sumy=scores_sumy['rouge1']
		rouge2_sumy=scores_sumy['rouge2']
		rougel_sumy=scores_sumy['rougeL']
		meteor_sumy = meteor_score([final_summary_sumy.split()], rawtext.split())
		bert_score_sumy=calculate_bert([final_summary_sumy], [rawtext])
		bert_score_sumy_p=bert_score_sumy[0]
		bert_score_sumy_r=bert_score_sumy[1]
		bert_score_sumy_f=bert_score_sumy[2]
	
		#BART		
		final_summary_bart = bart_summarizer(rawtext)
		summary_reading_time_bart = readingTime(final_summary_bart)	
		compress_ratio_bart=calculate_compression_ratio(rawtext,final_summary_bart)
		scores_bart=calculate_rouge(final_summary_bart,rawtext)
		rouge1_bart=scores_bart['rouge1']
		rouge2_bart=scores_bart['rouge2']
		rougel_bart=scores_bart['rougeL']
		meteor_bart = meteor_score([final_summary_bart.split()], rawtext.split())
		bert_score_bart=calculate_bert([final_summary_bart], [rawtext])
		bert_score_bart_p=bert_score_bart[0]
		bert_score_bart_r=bert_score_bart[1]
		bert_score_bart_f=bert_score_bart[2]

		#Pegasus	
		final_summary_pegasus = pegasus_summarizer(rawtext)
		summary_reading_time_pegasus = readingTime(final_summary_pegasus)
		compress_ratio_pegasus=calculate_compression_ratio(rawtext,final_summary_pegasus)
		scores_pegasus=calculate_rouge(final_summary_pegasus,rawtext)
		rouge1_pegasus=scores_pegasus['rouge1']
		rouge2_pegasus=scores_pegasus['rouge2']
		rougel_pegasus=scores_pegasus['rougeL']
		meteor_pegasus = meteor_score([final_summary_pegasus.split()], rawtext.split())
		bert_score_pegasus=calculate_bert([final_summary_pegasus], [rawtext])
		bert_score_pegasus_p=bert_score_pegasus[0]
		bert_score_pegasus_r=bert_score_pegasus[1]
		bert_score_pegasus_f=bert_score_pegasus[2]

		end = time.time()
		final_time = end-start
	return render_template('compare_summary.html',ctext=rawtext,final_summary_spacy=final_summary_spacy,
						   final_summary_nltk=final_summary_nltk,final_time=final_time,final_reading_time=final_reading_time,
						   summary_reading_time=summary_reading_time,final_summary_sumy=final_summary_sumy,
						   summary_reading_time_sumy=summary_reading_time_sumy,summary_reading_time_nltk=summary_reading_time_nltk,
						   final_summary_bart = final_summary_bart, summary_reading_time_bart=summary_reading_time_bart,
						   final_summary_pegasus = final_summary_pegasus, summary_reading_time_pegasus=summary_reading_time_pegasus,
						   compress_ratio_spacy=compress_ratio_spacy, compress_ratio_nltk=compress_ratio_nltk, compress_ratio_sumy=compress_ratio_sumy,
						   compress_ratio_bart=compress_ratio_bart, compress_ratio_pegasus=compress_ratio_pegasus,
						   rouge1_spacy=rouge1_spacy,rouge2_spacy=rouge2_spacy,rougel_spacy=rougel_spacy,
						   rouge1_nltk=rouge1_nltk,rouge2_nltk=rouge2_nltk,rougel_nltk=rougel_nltk,
						   rouge1_sumy=rouge1_sumy,rouge2_sumy=rouge2_sumy,rougel_sumy=rougel_sumy,
						   rouge1_bart=rouge1_bart,rouge2_bart=rouge2_bart,rougel_bart=rougel_bart,
						   rouge1_pegasus=rouge1_pegasus,rouge2_pegasus=rouge2_pegasus,rougel_pegasus=rougel_pegasus,
						   meteor_spacy=meteor_spacy, meteor_nltk=meteor_nltk, meteor_sumy=meteor_sumy,
						   meteor_bart=meteor_bart, meteor_pegasus=meteor_pegasus,
						   bert_score_spacy_p=bert_score_spacy_p,bert_score_spacy_r=bert_score_spacy_r,bert_score_spacy_f=bert_score_spacy_f,
						   bert_score_nltk_p=bert_score_nltk_p,bert_score_nltk_r=bert_score_nltk_r,bert_score_nltk_f=bert_score_nltk_f,
						   bert_score_sumy_p=bert_score_sumy_p,bert_score_sumy_r=bert_score_sumy_r,bert_score_sumy_f=bert_score_sumy_f,
						   bert_score_bart_p=bert_score_bart_p,bert_score_bart_r=bert_score_bart_r,bert_score_bart_f=bert_score_bart_f,
						   bert_score_pegasus_p=bert_score_pegasus_p,bert_score_pegasus_r=bert_score_pegasus_r,bert_score_pegasus_f=bert_score_pegasus_f)









@app.route('/about')
def about():
	return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)