import csv
import argparse
from fkgl import compute_fkgl
from sari import compute_sari
import pandas as pd
from nltk import word_tokenize


def spacy_tokenize(tokenizer, x):
    l_tokens = [doc.text for doc in tokenizer(x)]
    return ' '.join(l_tokens)

def print_metrics(complex_file, simplified_file, references_folder):
    df = pd.read_csv(simplified_file, sep='\n', quoting=csv.QUOTE_NONE, header=None)
    tknzd_simp_file = simplified_file + '.tknzd.tmp'
    pd.DataFrame(df[0].apply(lambda x: ' '.join(word_tokenize(x)))).to_csv(tknzd_simp_file, sep='\n', quoting=csv.QUOTE_NONE, header=None, index=False)

    slen, wlen, fkgl = compute_fkgl(tknzd_simp_file)
    print(f'Average Sentence Length : {slen:.4}')
    print(f'Average Word Length : {wlen:.4}')
    print(f'FKGL : {fkgl:.4}')
    print(f"===============\n")

    sari_score, sarif, add, keep, deletep, deleter, deletef = compute_sari(complex_file, references_folder, tknzd_simp_file)
    print(f'SARI score: {sari_score * 100:.4}')
    print(f'Add : {add * 100:.4}')
    print(f'Keep : {keep * 100:.4}')
    print(f'Delete : {deletep * 100:.4}')
    print("===============\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--complex", dest="complex", help="complex sentences", metavar="FILE")
    parser.add_argument("-r", "--reference", dest="reference", help="folder that contains files with references", metavar="FILE")
    parser.add_argument("-s", "--simplified", dest="simplified", help="simplified sentences. RAW data.", metavar="FILE")
    args = parser.parse_args()
    print_metrics(args.complex, args.simplified, args.reference)
