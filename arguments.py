import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--infile', type=str, help='input file')
parser.add_argument('--outfile', type=str, help='output file')
parser.add_argument('--rbt_path', type=str, required=True, help='path to finetuned multitask roberta')

parser.add_argument('--attribute', type=str, required=True, help='which attribute to revise. formality or simplicity')
parser.add_argument('--step_size', type=float, default=1.6, help='hyperparameter lambda')
parser.add_argument('--cls_thld', type=float, required=True, help='attrinbute threshold')
parser.add_argument('--C', type=float, default=1, help='smoothing constant')
parser.add_argument('--iter_step', type=int, default=4, help='maximum iteration')
parser.add_argument('--max_mask_ratio', type=float, default=0.3, help='the ratio of masked tokens in a sentence')
parser.add_argument('--fixed_span_len', type=int, default=3)

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=4)

args = parser.parse_args()
