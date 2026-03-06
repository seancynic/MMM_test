import os
import torch
import numpy as np

import options.option_transformer as option_trans
import models.vqvae as vqvae
from models.bitm import BiTMBERT

from utils.eval_bitm import inference_t2m
from exit.utils import generate_src_mask, visualize_2motions

from transformers import AutoTokenizer


def _pick_state_dict(ckpt: dict):
    """
    兼容不同保存格式：
    - train_bitm.py 加载用 ckpt['trans']
    - eval_bitm.py 保存用 {'bitm': state_dict}
    - 也可能有人直接 torch.save(state_dict)
    """
    if isinstance(ckpt, dict):
        for k in ["trans", "bitm", "state_dict", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


class MMM_BiTM(torch.nn.Module):
    def __init__(self, args, bert_name="google-bert/bert-large-uncased"):
        super().__init__()
        self.args = args
        self.bert_name = bert_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------- tokenizer (必须和训练 bitm 时一致) ----------
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_name)

        # ---------- VQ-VAE ----------
        self.vqvae = vqvae.HumanVQVAE(
            args,
            args.nb_code,
            args.code_dim,
            args.output_emb_width,
            args.down_t,
            args.stride_t,
            args.width,
            args.depth,
            args.dilation_growth_rate,
        )
        if args.resume_pth is None:
            raise ValueError("args.resume_pth 为空：请传 VQ-VAE 的 net_last.pth 路径")
        ckpt_vq = torch.load(args.resume_pth, map_location="cpu")
        self.vqvae.load_state_dict(ckpt_vq["net"], strict=True)
        self.vqvae.eval().to(self.device)

        # motion special ids（和 train_bitm.py 一致）
        self.special_ids_m = {
            "end_id": args.nb_code,        # end
            "pad_id": args.nb_code + 1,    # pad
            "mask_id": args.nb_code + 2,   # mask
        }

        # ---------- BiTMBERT ----------
        self.bitm = BiTMBERT(
            bert_name=self.bert_name,
            vqvae=self.vqvae,
            vocab_m=args.nb_code,
            max_t=args.max_t,
            max_m=args.max_m,
            first_modality=args.first_modality,
            dropout_rate=args.drop_out_rate,
        )
        if args.resume_trans is None:
            raise ValueError("args.resume_trans 为空：请传 bitm 的 checkpoint 路径（例如 net_last.pth / net_last_t.pth）")
        ckpt_bitm = torch.load(args.resume_trans, map_location="cpu")
        state = _pick_state_dict(ckpt_bitm)
        self.bitm.load_state_dict(state, strict=True)
        self.bitm.eval().to(self.device)

    @torch.no_grad()
    def forward(self, texts, lengths, rand_pos=False, max_steps=10):
        """
        texts: List[str] 或 str
        lengths: torch.Tensor (frames) 或 int/list[int]
        rand_pos=False -> temperature=0 (更确定性)
        """
        if isinstance(texts, str):
            texts = [texts]
        bs = len(texts)

        if isinstance(lengths, (int, np.integer)):
            lengths = torch.tensor([int(lengths)], dtype=torch.long)
        elif isinstance(lengths, (list, tuple)):
            lengths = torch.tensor(list(lengths), dtype=torch.long)
        lengths = lengths.to(self.device)

        # 1 token 对应多少帧：unit_length = 2 ** down_t（t2m 默认 down_t=2 -> 4帧）
        unit = 2 ** self.args.down_t
        lens_m = torch.ceil(lengths.float() / unit).long()

        # 保护：lens_m 最大只能到 max_m-1（因为 index=lens_m 处要放 end_id）
        lens_m = lens_m.clamp(min=1, max=self.args.max_m - 1)

        # tokenize text -> token_ids_t / seq_mask_t
        text_inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.args.max_t,
            return_tensors="pt",
        )
        token_ids_t = text_inputs["input_ids"].to(self.device)         # (bs, max_t)
        seq_mask_t = text_inputs["attention_mask"].to(self.device)     # (bs, max_t)

        # motion mask
        seq_mask_m = generate_src_mask(self.args.max_m, lens_m + 1).to(self.device)     # include END
        seq_mask_no_end_m = generate_src_mask(self.args.max_m, lens_m).to(self.device) # exclude END

        # sample motion tokens with iterative masking
        index_motion = inference_t2m(
            model=self.bitm,
            lens_m=lens_m,
            token_ids_t=token_ids_t,
            seq_mask_t=seq_mask_t,
            seq_mask_m=seq_mask_m,
            seq_mask_no_end_m=seq_mask_no_end_m,
            special_ids_m=self.special_ids_m,
            max_length=self.args.max_m - 1,
            shape=(bs, self.args.max_m),
            rand_pos=rand_pos,
            max_steps=max_steps,
        )  # (bs, max_m)

        # decode motion tokens -> pose seq
        dim_m = 251 if self.args.dataname == "kit" else 263
        max_len = int(lengths.max().item())
        pred_pose_all = torch.zeros((bs, max_len, dim_m), device=self.device)

        for k in range(bs):
            tok = index_motion[k:k + 1, : lens_m[k].item()]  # 不包含 end_id
            pred_pose = self.vqvae(tok, type="decode")       # (1, T, dim_m)
            pred_pose_all[k:k + 1, : lengths[k].item()] = pred_pose[:, : lengths[k].item()]

        return pred_pose_all, index_motion, lens_m


if __name__ == "__main__":
    args = option_trans.get_args_parser()

    # 保险：与你原 MMM 脚本一致
    args.dataname = getattr(args, "dataname", "t2m")
    args.dataset_name = args.dataname

    # 推理
    mmm = MMM_BiTM(args).to("cuda" if torch.cuda.is_available() else "cpu")
    pred_pose, index_motion, lens_m = mmm(args.text, args.length, rand_pos=False, max_steps=10)

    # 可视化（沿用你原来的 mean/std）
    if args.dataname == "t2m":
        std = np.load("./exit/t2m-std.npy")
        mean = np.load("./exit/t2m-mean.npy")
    else:
        # 如果你在 KIT 上跑，请把下面两行改成你项目里对应的 mean/std 文件
        std = np.load("./exit/kit-std.npy")
        mean = np.load("./exit/kit-mean.npy")

    os.makedirs("./output", exist_ok=True)
    file_name = "_".join(args.text.split(" ")) + "_" + str(args.length)
    save_path = f"./output/{file_name}_bitm.html"

    visualize_2motions(
        pred_pose[0].detach().cpu().numpy(),
        std,
        mean,
        args.dataname,
        args.length,
        save_path=save_path,
    )
    print(f"[OK] Saved to: {save_path}")
    print(f"[DBG] lens_m(token_len)={lens_m[0].item()}, first 10 tokens={index_motion[0, :10].tolist()}")