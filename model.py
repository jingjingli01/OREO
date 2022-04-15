import torch
import torch.nn as nn

ABBR = ("'s","'S","'t","'t","'re","'RE","'ve","'VE","'m","'M","'ll","'LL","'d","'D") # abbreviation

class OREO():
    def __init__(self, rbt_model, rbt_tknzr, device, k):
        self.model = rbt_model
        self.tokenizer = rbt_tknzr
        self.device = device
        self.K = k
        self.vocab_size = len(self.tokenizer)

        self.mask_idx = self.tokenizer.convert_tokens_to_ids(['<lm-mask>'])[0]
        self.pad_idx = self.tokenizer.convert_tokens_to_ids(['<pad>'])[0]
        self.bos_idx = self.tokenizer.convert_tokens_to_ids(['<s>'])[0]
        self.eos_idx = self.tokenizer.convert_tokens_to_ids(['</s>'])[0]

        self.model.eval()
        self.model.to(self.device)

    def cal_attr(self, cand_inps, hook_hid_grad):
        attn_mask = cand_inps.ne(self.pad_idx).float().to(self.device)
        attr_val, hid_states = self.model('cls', cand_inps,
                                            attention_mask=attn_mask,
                                            hook_hid_grad=hook_hid_grad)
        return attr_val, hid_states
    
    def select_abbr_span(self, tknzd_sents):
        abbr_pos = []
        for sent in tknzd_sents:
            pos = []
            for i, tk in enumerate(sent):
                if tk in ABBR:
                    pos.extend([i, i + 1])
                    break
            abbr_pos.append(pos)
        return abbr_pos

    def revise(self, K, cand_inps, cand_mask, ins_pos, memory_bank=None):
        bsz, seqlen = cand_inps.size()

        t = 0

        with torch.no_grad():
            editable = cand_mask.float()

            while editable.eq(1).any():
                if t == 0:
                    attn_mask = cand_inps.ne(self.pad_idx).type(torch.float)
                    outs = self.model('maskedlm', cand_inps, attention_mask=attn_mask,
                                      memory_bank=memory_bank, memory_fix_pos=ins_pos)[0]
                    outs = outs.transpose(1, 2).contiguous().view(bsz, -1)
                    outs = torch.where(editable.repeat(1, self.vocab_size).eq(1).to(self.device),
                                       outs, torch.full_like(outs, -1e10))
                    ins_probs, cand_words = torch.topk(outs, K, dim=-1)
                    del outs, ins_probs

                    edit_pos_t = cand_words % seqlen
                    cand_words = cand_words // seqlen
                    cand_inps = cand_inps.repeat(1, K).contiguous().view(bsz * K, seqlen)
                    editable = editable.repeat(1, K).contiguous().view(bsz * K, seqlen)
                    edit_pos_t = edit_pos_t.view(-1, 1)
                    cand_words = cand_words.view(-1, 1)
                else:
                    cand_words_all = []
                    edit_pos_t = []
                    cand_inps_bats = torch.split(cand_inps, 100, dim=0)
                    editable_bats = torch.split(editable, 100, dim=0)
                    for b, (inps, editable_t) in enumerate(zip(cand_inps_bats, editable_bats)):
                        outs = self.model('maskedlm', inps.to(self.device),
                                          attention_mask=attn_mask, memory_bank=memory_bank,
                                          memory_fix_pos=ins_pos)[0]
                        outs = outs.transpose(1, 2).contiguous().view(outs.size(0), -1)
                        outs = torch.where(editable_t.repeat(1, self.vocab_size).eq(1).to(self.device),
                                           outs, torch.full_like(outs, -1e10))
                        ins_probs, cand_words = torch.topk(outs, 1, dim=-1)
                        del outs, ins_probs, editable_t
                        edit_pos_tb = cand_words % seqlen
                        cand_words = cand_words // seqlen
                        cand_words_all.append(cand_words)
                        edit_pos_t.append(edit_pos_tb)
                    cand_words = torch.cat(cand_words_all, dim=0)
                    edit_pos_t = torch.cat(edit_pos_t, dim=0)
                t += 1
                assert cand_words.ne(self.mask_idx).all()
                new_cand_inps = cand_inps.scatter(1, edit_pos_t, cand_words)

                cand_inps = torch.where(editable.eq(1), new_cand_inps, cand_inps)

                del new_cand_inps
                edit_pos_t = edit_pos_t.view(-1, 1)
                editable = editable.scatter(1, edit_pos_t.to(self.device),
                                            torch.zeros_like(edit_pos_t).float().to(self.device))
        return cand_inps

