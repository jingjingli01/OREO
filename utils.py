import torch
import random
import en_core_web_lg
import en_core_web_sm

if torch.cuda.is_available():
    nlp = en_core_web_lg.load()
else:
    nlp = en_core_web_sm.load() # for cpu

CAP_TYPES = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'EVENT', 'WORK_OF_ART', 'LANGUAGE', 'LAW']

ENT_TYPES = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'EVENT', 'WORK_OF_ART', 'LANGUAGE', 'LAW', 'PERCENT', 'TIME', ]


def batchify(iter, bsz):
    for i in range(0, len(iter), bsz):
        yield iter[i: i+bsz]


def padding(arr, pad_token, dtype=torch.long, mask_head_tail=False):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        if mask_head_tail:
            mask[i, 1:lens[i] - 1] = 1
        else:
            mask[i, :lens[i]] = 1
    return padded, lens, mask


def pad_to_fix_len(arr, pad_token, maxlen, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = maxlen
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask


def collate_no_tokenize(arr, numericalize, fixed_span_len=None,
                           pad="<pad>", device='cuda',
                           ins_pos_l=None, pad_inp=False,):
    arr = [["<s>"] + a + ["</s>"] for a in arr]
    arr = [numericalize(a) for a in arr]
    
    pad_token = numericalize([pad])[0]
    plh_token = numericalize(['<plh>'])[0]
    
    sent_indices, pad_ins_pos = [], []
    if ins_pos_l:
        if pad_inp:
            for i, pos in enumerate(ins_pos_l):
                n_plh = fixed_span_len - len(pos)
                arr[i] = arr[i][:pos[-1] + 1] + [plh_token] * n_plh + arr[i][pos[-1] + 1:]
                pad_ins_pos.extend(list(range(pos[0], pos[0] + fixed_span_len)))
                sent_indices.extend([i] * fixed_span_len)
        else:
            for i, pos in enumerate(ins_pos_l):
                sent_indices.extend([i] * len(pos))
                pad_ins_pos.extend(list(range(pos[0], pos[-1] + 1)))
    
    ins_pos = [torch.tensor(sent_indices), torch.tensor(pad_ins_pos)]
    
    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    
    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, lens, mask, ins_pos


def collate_inp_mask_after_span(arr, numericalize, ins_pos_l,
                           pad="<pad>", device='cuda'):
    arr = [["<s>"] + a + ["</s>"] for a in arr]
    arr = [numericalize(a) for a in arr]
    
    pad_token = numericalize([pad])[0]
    mask_token = numericalize(['<lm-mask>'])[0]

    sent_indices, pad_ins_pos = [], []
    for i, pos in enumerate(ins_pos_l):
        arr[i] = arr[i][:pos[-1] + 1] + [mask_token] + arr[i][pos[-1] + 1:]
        pad_ins_pos.extend(list(range(pos[0], pos[-1] + 2)))
        sent_indices.extend([i] * (pos[-1] + 2 - pos[0]))
    ins_pos = [torch.tensor(sent_indices).to(device), torch.tensor(pad_ins_pos).to(device)]
    
    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)

    return padded.to(device), lens.to(device), mask.to(device), ins_pos


def collate_cls_inp(arr, numericalize, fix_len, pad="<pad>", device='cuda'):
    arr = [["<s>"] + a + ["</s>"] for a in arr]
    arr = [numericalize(a) for a in arr]
    
    pad_token = numericalize([pad])[0]
    
    sent_indices, pad_ins_pos = [], []
    
    ins_pos = [torch.tensor(sent_indices), torch.tensor(pad_ins_pos)]
    
    padded, lens, mask = pad_to_fix_len(arr, pad_token, maxlen=fix_len, dtype=torch.long)
    
    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, lens, mask, ins_pos


def ids2sents(ids_tensor, tknz, lens, rm_plh=True):
    sents = []
    lens = lens.cpu().tolist()
    for i, ids in enumerate(ids_tensor.cpu().tolist()):
        ids = ids[1:lens[i]-1]
        sent = tknz.convert_ids_to_tokens(ids)
        if rm_plh:
            while '<plh>' in sent:
                sent.remove('<plh>')
        sents.append(sent)
    return sents


def get_ngram_topk(inps_grad, mask, C, max_span_len, fixed_span_len=3):
    inps_grad = torch.where(mask.eq(1), inps_grad, torch.full_like(inps_grad, -1e16))
    bsz, seqlen = inps_grad.size()
    
    ngrams_b = [[] for _ in range(bsz)]
    for ngram in range(1, fixed_span_len + 1):
        ngram_grad = 0.
        for i in range(ngram):
            ngram_grad += inps_grad[:, i: i + (seqlen - ngram + 1)]
        
        value, position = torch.topk(ngram_grad / (ngram + C), 1, dim=1)
        for bat, (val, pos) in enumerate(zip(value.cpu().tolist(), position.cpu().tolist())):
            for v, p in zip(val, pos):
                if v > 0:
                    ngrams_b[bat].append((v, list(range(p, p + ngram))))
                else:
                    assert ValueError, "Value of topk grad cannot be negative"

    sorted_ngrams_b = []
    for bat, seq in enumerate(ngrams_b):
        seq = [val_idx for val_idx in seq if len(val_idx[1]) <= max_span_len[bat]]
        sorted_ngrams_b.append(sorted(seq, key=lambda x: x[0], reverse=True)[0][1])

    return sorted_ngrams_b


def get_rand_span(mask, max_span_len, fixed_span_len=3):
    rand_span_pos = []
    bsz, seqlen = mask.size()
    playground = torch.arange(seqlen, dtype=mask.dtype).repeat(bsz, 1).cuda()
    playground = torch.where(mask.eq(1), playground, mask)
    for i in range(bsz):
        playground_l = playground[i, :].cpu().tolist()
        non_ner_pos = [item for item in playground_l if item > 0]
        not_find_legal = True
        while not_find_legal:
            span_len = min(random.randint(1, fixed_span_len), int(max_span_len[i].item()))
            span_start = random.choice(non_ner_pos)
            span_pos = list(range(span_start, span_start + span_len))

            if sum(span_pos) != sum(playground_l[span_start: span_start + span_len]):
                not_find_legal = True
            else:
                not_find_legal = False
                rand_span_pos.append(span_pos)
    return rand_span_pos


def get_entity_mask(sents, sents_id, tokenizer, add_prefix_space):
    def spanfinder(sent_id, span):
        mask = torch.ones_like(sent_id)
        for i in range(len(sent_id)):
            if sent_id[i] == span[0]:
                if sent_id[i:i+len(span)].tolist() == span:
                    mask[i:i+len(span)] = 0
                    
        return mask
    
    ent_mask = torch.ones_like(sents_id)
    docs = list(nlp.pipe(sents, disable=["tagger", "parser"]))

    for i, doc in enumerate(docs):
        for ent in doc.ents:
            if ent.label_ in ENT_TYPES:
                if ent.start_char == 0:
                    ent_id_span = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ent.text, add_prefix_space=add_prefix_space))
                else:
                    ent_id_span = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ent.text, add_prefix_space=True))
                ent_mask[i] = spanfinder(sents_id[i], ent_id_span)
    return ent_mask


def capital_ne(sents):
    docs = list(nlp.pipe(sents, disable=["tagger", "parser"]))

    for id, doc in enumerate(docs):
        for ent in doc.ents:
            if ent.label_ in CAP_TYPES:
                sents[id] = sents[id].replace(ent.text, ent.text.title())
    return sents


def transform_ins_pos_l(ins_pos_l, bsz, device, pad_num):
    max_ins_num = max(len(item) for item in ins_pos_l) + pad_num
    ins_pos_mat = torch.zeros(max_ins_num, bsz).long()
    for i, item in enumerate(ins_pos_l):
        padded_item = list(range(item[0], item[-1] + 1 + pad_num))
        ins_pos_mat[:, i][:len(padded_item)] = torch.tensor(padded_item, dtype=torch.long)
    return torch.split(ins_pos_mat.to(device), 1, dim=0)
