import csv
import os
import sys
import inspect
import argparse
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import numpy as np
import torch
import random
from sklearn.metrics import f1_score

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from mtl_roberta.modeling_roberta import RobertaConfig, RobertaForMTL
from mtl_roberta.tokenization_roberta import RobertaTokenizer
from utils import capital_ne, collate_no_tokenize, batchify


def cal_bleu(pred_df, ref_dir):
	def cal_sent_bleu(ref1, ref2, ref3, ref4, pred):
		return sentence_bleu([ref1, ref2, ref3, ref4], pred, \
							 smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)

	pred_l = len(pred_df)
	refs = []
	for i in range(4):
		file = os.path.join(ref_dir, f"formal.ref{i}")
		df = pd.read_csv(file, sep='\n', quoting=csv.QUOTE_NONE, header=None, names=[f'ref{i}',])
		assert len(df) == pred_l
		df = df[f'ref{i}'].apply(lambda x: word_tokenize(x.lower()))
		refs.append(df)
	all_df = pd.concat(refs, axis=1)
	all_df['pred'] = pred_df[0].apply(lambda x: word_tokenize(x.lower()))
	all_df['bleu'] = list(map(cal_sent_bleu, all_df['ref0'], all_df['ref1'], all_df['ref2'], all_df['ref3'], all_df['pred']))
	bleu_score = np.mean(np.array(all_df['bleu']))
	return bleu_score * 100


def cal_attr(model_path, preds, device, bsz):
	config = RobertaConfig.from_pretrained(model_path, cache_dir=None)
	tokenizer = RobertaTokenizer.from_pretrained(model_path, do_lower_case=False, cache_dir=None)
	model = RobertaForMTL.from_pretrained(model_path, config=config, task_names=['cls'])
	model.to(device)
	model.eval()
	
	pred_attr = None
	for sents in batchify(preds, bsz):
		inps = [tokenizer.tokenize(sent, add_prefix_space=True) for sent in sents]
		input_ids, _, input_mask, _ = collate_no_tokenize(inps, \
                                         tokenizer.convert_tokens_to_ids, device=device)
		with torch.no_grad():
			logits = model('cls', input_ids=input_ids, attention_mask=input_mask)[0]

		if pred_attr is None:
			pred_attr = logits.detach().cpu().numpy()
		else:
			pred_attr = np.append(pred_attr, logits.detach().cpu().numpy(), axis=0)
			
	pred_attr = np.argmax(pred_attr, axis=1)
	labels = np.ones_like(pred_attr)
	acc = (pred_attr == labels).mean()
	f1 = f1_score(y_true=labels, y_pred=pred_attr)
	result = {'acc': acc * 100, 'f1': f1 * 100}
	return result


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--result_file', type=str)
	parser.add_argument('--outfile', type=str)
	parser.add_argument('--ref_dir', type=str)
	parser.add_argument('--cls_model', type=str)
	parser.add_argument('--lm_model', type=float)
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--batch_size', type=int, default=10)
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	df = pd.read_csv(args.result_file, sep='\n', quoting=csv.QUOTE_NONE, header=None)
	bleu_score = cal_bleu(df, args.ref_dir)

	df[1] = df[0].apply(lambda x: x.replace(' i ', ' I ')).apply(lambda x: x[0].upper() + x[1:])
	cased_preds = capital_ne(df[1].tolist())
	cls_score = cal_attr(args.cls_model, cased_preds, device, args.batch_size)

	print(f"Results: bleu_score = {bleu_score: .2f}, cls_score = {cls_score['acc']: .2f}")


if __name__ == '__main__':
	main()
