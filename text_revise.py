import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import csv

from mtl_roberta.modeling_roberta import RobertaConfig, RobertaForMTL
from mtl_roberta.tokenization_roberta import RobertaTokenizer
from utils import get_entity_mask, collate_no_tokenize, ids2sents, \
    batchify, get_ngram_topk, collate_inp_mask_after_span
from model import OREO
from arguments import args


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    K = 1
    rbt_config = RobertaConfig.from_pretrained(args.rbt_path, cache_dir=None)
    rbt_tknzr = RobertaTokenizer.from_pretrained(args.rbt_path, do_lower_case=False)
    rbt_model = RobertaForMTL.from_pretrained(args.rbt_path, config=rbt_config,
                                              task_names=['bertscore', 'maskedlm', 'cls'])
    rbt_model.to(device)
    oreo = OREO(rbt_model=rbt_model, rbt_tknzr=rbt_tknzr, k=K, device=device)
    data = pd.read_csv(args.infile, sep='\n', quoting=csv.QUOTE_NONE, header=None)
    results = []

    for orgs in tqdm(batchify(data[0].tolist(), args.batch_size), total=len(data[0])/args.batch_size, desc='loading input'):
        torch.cuda.empty_cache()
        cand_inps_sents = orgs.copy()
        edit_track = [[] for _ in range(len(orgs))]

        for step in range(args.iter_step):
            torch.cuda.empty_cache()

            inps = [rbt_tknzr.tokenize(sent, add_prefix_space=True)[:50] for sent in cand_inps_sents]

            if args.attribute == 'formality':
                abbr_pos = oreo.select_abbr_span(inps)

            cand_inps, cand_lens, _, _ = collate_no_tokenize(inps, rbt_tknzr.convert_tokens_to_ids, device=device)
            bsz, seqlen = cand_inps.size()
            oreo.model.output_hidden_states = True
            attr_val_org, hid_states_org = oreo.cal_attr(cand_inps, hook_hid_grad=True)
            oreo.model.output_hidden_states = False

            loss = F.cross_entropy(attr_val_org, torch.ones(bsz).long().to(device))
            loss.backward()

            attr_val = F.softmax(attr_val_org, dim=1).cpu()
            del attr_val_org
            attr_val_mask = torch.where(attr_val[:, 1] > args.cls_thld, torch.zeros(bsz),
                                        torch.ones(bsz)).bool().tolist()

            attr_scores = attr_val[:, 1].tolist()
            for i, (sent, attr_score) in enumerate(zip(cand_inps_sents, attr_scores)):
                ex = {'sent': sent, 'score': attr_score}
                edit_track[i].append(ex)

            ent_mask = get_entity_mask(cand_inps_sents, cand_inps, rbt_tknzr, add_prefix_space=True)
            grad_mask = (cand_inps.ne(oreo.pad_idx)) * (cand_inps.ne(oreo.bos_idx)) * \
                        (cand_inps.ne(oreo.eos_idx)) * ent_mask

            del cand_inps, ent_mask

            for i, state in enumerate(hid_states_org):
                norm = torch.norm(state.grad, dim=-1).unsqueeze(2)
                norm = torch.where(norm > 0, norm, torch.full_like(norm, 1e-10))
                if i == 0:
                    max_span_len = torch.floor((cand_lens - 2) * args.max_mask_ratio)
                    max_span_len = torch.where(max_span_len < 1, torch.ones_like(max_span_len), max_span_len)
                    emb_ngram_top1 = get_ngram_topk(norm.squeeze(-1), grad_mask, args.C, max_span_len, args.fixed_span_len)
                    break

            del grad_mask, hid_states_org
            oreo.model.zero_grad()

            if args.attribute == 'formality':
                ins_pos_l = [item if item else emb_ngram_top1[i] for i, item in enumerate(abbr_pos)]
            elif args.attribute == 'simplicity':
                ins_pos_l = emb_ngram_top1

            cand_inps, cand_lens, _, ins_pos = collate_inp_mask_after_span(inps, rbt_tknzr.convert_tokens_to_ids,
                                                                           ins_pos_l,
                                                                           device=device)
            bsz, seqlen = cand_inps.size()
            oreo.model.output_hidden_states = True
            attr_val_org, hid_states_pad = oreo.cal_attr(cand_inps, hook_hid_grad=True)
            oreo.model.output_hidden_states = False

            loss = F.cross_entropy(attr_val_org, torch.ones(bsz).long().to(device))
            loss.backward()
            del attr_val_org

            cand_ins_inps = cand_inps.index_put(ins_pos, torch.tensor(oreo.mask_idx).to(device))

            for i, state in enumerate(hid_states_pad):
                if i == 0: continue
                norm = torch.norm(state.grad, dim=-1).unsqueeze(2)
                norm = torch.where(norm > 0, norm, torch.full_like(norm, 1e-10))
                hid_states_pad[i] = state - args.step_size * state.grad / norm

            cand_mask = cand_ins_inps.eq(oreo.mask_idx)
            cand_inps = oreo.revise(K, cand_ins_inps, cand_mask, \
                                    ins_pos, memory_bank=hid_states_pad[1:])
            del hid_states_pad
            oreo.model.zero_grad()

            cand_inps = cand_inps[:bsz]
            mid_cand_inps_toks = ids2sents(cand_inps.view(-1, cand_inps.size(-1)), rbt_tknzr, cand_lens)
            edited_cand_inps_sents = [rbt_tknzr.convert_tokens_to_string(x).lstrip() for x in mid_cand_inps_toks]

            if args.attribute == 'formality':
                for i, (val, has_abbr) in enumerate(zip(attr_val_mask, abbr_pos)):
                    if has_abbr or val:
                        cand_inps_sents[i] = edited_cand_inps_sents[i]
            elif args.attribute == 'simplicity':
                for i, val in enumerate(attr_val_mask):
                    if val:
                        cand_inps_sents[i] = edited_cand_inps_sents[i]

            if step == args.iter_step - 1:
                tknzd_inps = [rbt_tknzr.tokenize(sent, add_prefix_space=True) for sent in cand_inps_sents]
                cand_inps, cand_lens, _, _ = collate_no_tokenize(tknzd_inps, rbt_tknzr.convert_tokens_to_ids, device=device)
                bsz, seqlen = cand_inps.size()
                oreo.model.output_hidden_states = True
                with torch.no_grad():
                    attr_val_org = oreo.cal_attr(cand_inps, hook_hid_grad=False)[0]
                    attr_scores = F.softmax(attr_val_org, dim=1)[:, 1].cpu().tolist()
                oreo.model.output_hidden_states = False
                del attr_val_org, cand_inps
                for i, (sent, attr_score) in enumerate(zip(cand_inps_sents, attr_scores)):
                    ex = {'sent': sent, 'score': attr_score}
                    edit_track[i].append(ex)

            edit_spans = []
            for i, tknz_sent in enumerate(inps):
                span = ins_pos_l[i]
                topk_tokens = tknz_sent[span[0] - 1: span[-1]]
                edit_spans.append((topk_tokens, span))

        for sent in edit_track:
            results.append(sorted(sent, key=lambda x: x['score'])[-1]['sent'])

    df = pd.DataFrame(results)
    outpath, filename = os.path.split(args.outfile)
    os.makedirs(outpath, exist_ok=True)
    df.to_csv(args.outfile, sep='\n', quoting=csv.QUOTE_NONE, header=None, index=False)
