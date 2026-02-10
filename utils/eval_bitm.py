import os
import torch
import numpy as np
from scipy import linalg
from tqdm import tqdm
from einops import rearrange
from exit.utils import get_model, generate_src_mask, cosine_schedule, gumbel_sample

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

try:
    from nlgmetricverse import NLGMetricverse, load_metric
except ImportError:
    NLGMetricverse = None
    load_metric = None

try:
    # the bert_score package exposes a `score` function
    from bert_score import score as bert_score
except ImportError:
    bert_score = None

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

##### ---- T2M Evaluations ---- #####
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)  # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)  # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        # print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat

def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()

def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0),
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0),
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist

def inference_t2m(model, lens_m: torch.Tensor, token_ids_t, seq_mask_t, seq_mask_m, seq_mask_no_end_m,
             special_ids_m, max_length, shape, rand_pos=True, token_cond=None, max_steps=10):
    # init sampling score
    scores = torch.ones(shape, dtype=torch.float32, device=lens_m.device)

    # init motion token ids
    if token_cond is not None:  # has partial condition
        token_ids_m = token_cond.clone()
        token_ids_m[~seq_mask_no_end_m] = special_ids_m['pad_id']
        num_token_cond = (token_ids_m == special_ids_m['mask_id']).sum(-1)
    else:  # start from full mask
        token_ids_m = torch.full(shape, special_ids_m['mask_id'], dtype=torch.long, device=lens_m.device)

    sample_max_steps = torch.round(max_steps / max_length * lens_m) + 1e-8
    for step in range(max_steps):
        timestep = torch.clip(step / sample_max_steps, max=1)
        if len(lens_m) == 1 and step > 0 and torch.clip((step - 1) / sample_max_steps, max=1).cpu().item() == timestep:
            break
        rand_mask_prob = cosine_schedule(timestep)
        num_token_masked = (rand_mask_prob * lens_m).long().clip(min=1)

        if token_cond is not None:
            num_token_masked = (rand_mask_prob * num_token_cond).long().clip(min=1)
            scores[token_cond != special_ids_m['mask_id']] = 0

        # remove no motion frames
        scores[~seq_mask_no_end_m] = 0
        scores = scores / scores.sum(-1)[:, None]  # normalize only unmasked token

        _, sorted_score_indices = scores.sort(descending=True)  # deterministic

        token_ids_m[~seq_mask_m] = special_ids_m['pad_id']  # replace with pad id
        token_ids_m.scatter_(-1, lens_m[..., None].long(), special_ids_m['end_id'])  # replace with end id

        # replace "mask_id" to "ids" that have highest "num_token_masked" "scores"
        select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked)
        # repeat last_id to make it scatter_ the existing last ids
        last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1) - 1)
        sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index * ~select_masked_indices)
        token_ids_m.scatter_(-1, sorted_score_indices, special_ids_m['mask_id'])

        logits = model(token_ids_t, token_ids_m, seq_mask_t, seq_mask_m)

        if rand_pos:
            temperature = 1  # starting_temperature * (steps_until_x0 / timesteps)  # temperature is annealed
        else:
            temperature = 0  # starting_temperature * (steps_until_x0 / timesteps)  # temperature is annealed

        # if temperature == 0, it is equal to argmax (pred_ids = pred_m.argmax(dim=-1))
        pred_ids_m = gumbel_sample(logits['logits_m'], temperature=temperature, dim=-1)
        is_mask = token_ids_m == special_ids_m['mask_id']

        token_ids_m = torch.where(is_mask, pred_ids_m, token_ids_m)

        # Update score
        probs_without_temperature = logits['logits_m'].softmax(dim=-1)
        scores = 1 - probs_without_temperature.gather(-1, pred_ids_m[..., None])
        scores = rearrange(scores, '... 1 -> ...')
        scores = scores.masked_fill(~is_mask, 0)

    return token_ids_m

@torch.no_grad()
def eval_bitm_t2m(out_dir, val_loader, net, bitm, logger, writer, nb_iter, eval_wrapper, special_ids_m, max_m,
                  best_iter=0, best_fid=1000, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100,
                  draw=True, save=True, num_repeat=1, rand_pos=False):
    if num_repeat < 0:  # evaluate all generations
        is_avg_all = True
        num_repeat = -num_repeat
    else:  # evaluate last generation
        is_avg_all = False

    bitm.eval()
    nb_sample = 0
    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0.
    R_precision = 0.
    matching_score_real = 0.
    matching_score_pred = 0.

    for batch in tqdm(val_loader, position=1, leave=True):
        word_embeddings, pos_one_hots, sent_len, _, pose, m_length, token_ids_t, seq_mask_t, _ = batch
        bs, seq = pose.shape[:2]
        lens_m = torch.ceil(m_length / 4).long()
        pred_len = m_length.cuda()

        # generate target token masks
        seq_mask_m = generate_src_mask(max_m, lens_m + 1)
        seq_mask_no_end_m = generate_src_mask(max_m, lens_m)

        motion_multimodality_batch = []

        for i in range(num_repeat):
            index_motion = inference_t2m(bitm,
                                         lens_m=lens_m.cuda(),
                                         token_ids_t=token_ids_t.cuda(),
                                         seq_mask_t=seq_mask_t.cuda(),
                                         seq_mask_m=seq_mask_m.cuda(),
                                         seq_mask_no_end_m=seq_mask_no_end_m.cuda(),
                                         special_ids_m=special_ids_m,
                                         max_length=max_m - 1,
                                         shape=(bs, max_m),
                                         rand_pos=rand_pos)  # (bs, max_m)

            # [INFO] need to run single sample at a time because it's conv
            pred_pose_eval = torch.zeros(pose.shape).cuda()
            for k in range(bs):
                # [INFO] Eval by m_length
                pred_pose = net(index_motion[k:k + 1, :lens_m[k].item()], type='decode')  # (1, m_length, dim_m)
                pred_pose_eval[k:k + 1, :pred_len[k].item()] = pred_pose  # (bs, m_length, dim_m)

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))

            if i == 0 or is_avg_all:
                pose = pose.cuda().float()

                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match

                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    if num_repeat > 1:
        multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, \n\
                FID. {fid:.4f} , \n\
                Diversity Real. {diversity_real:.4f}, \n\
                Diversity. {diversity:.4f}, \n\
                R_precision_real. {R_precision_real}, \n\
                R_precision. {R_precision}, \n\
                matching_score_real. {matching_score_real}, \n\
                matching_score_pred. {matching_score_pred}, \n\
                multimodality. {multimodality:.4f}"
    logger.info(msg)

    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)
        writer.add_scalar('./Test/multimodality', multimodality, nb_iter)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter

    if matching_score_pred < best_matching:
        msg = f"--> --> \t Matching Score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if save:
        torch.save({'bitm': get_model(bitm).state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    bitm.train()
    return best_iter, best_fid, best_div, best_top1, best_top2, best_top3, best_matching, multimodality

##### ---- M2T Evaluations ---- #####
def decode_token_ids(token_ids, tokenizer, eos_id):
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()

    batch_tokens = []
    for row in token_ids:
        if eos_id in row:
            batch_tokens.append(row[:row.index(eos_id)])
        else:
            batch_tokens.append(row)

    return tokenizer.batch_decode(batch_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def compute_bleu_scores(preds, refs):
    bleu1, bleu4 = 0., 0.
    for pred, ref in zip(preds, refs):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        bleu1 += sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0))
        bleu4 += sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))

    return {
        "BLEU-1": bleu1 / len(preds),
        "BLEU-4": bleu4 / len(preds),
    }

def compute_rouge_l(preds, refs, scorer=None):
    if scorer is None:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    rouge_l_f = 0.
    for pred, ref in zip(preds, refs):
        score = scorer.score(ref, pred)
        rouge_l_f += score['rougeL'].fmeasure

    return rouge_l_f / len(preds)

def inference_m2t(model, lens_t: torch.Tensor, token_ids_m, seq_mask_m, seq_mask_t, seq_mask_no_end_t,
             special_ids_t, max_length, shape, rand_pos=True, token_cond=None, max_steps=10):
    # init sampling score
    scores = torch.ones(shape, dtype=torch.float32, device=lens_t.device)

    # init text token ids
    if token_cond is not None:  # has partial condition
        token_ids_t = token_cond.clone()
        token_ids_t[~seq_mask_no_end_t] = special_ids_t['pad_id']
        num_token_cond = (token_ids_t == special_ids_t['mask_id']).sum(-1)
    else:  # start from full mask
        token_ids_t = torch.full(shape, special_ids_t['mask_id'], dtype=torch.long, device=lens_t.device)
        token_ids_t[:, 0] = special_ids_t['cls_id']  # add [CLS] token for text

    sample_max_steps = torch.round(max_steps / max_length * lens_t) + 1e-8
    for step in range(max_steps):
        timestep = torch.clip(step / sample_max_steps, max=1)
        if len(lens_t) == 1 and step > 0 and torch.clip((step - 1) / sample_max_steps, max=1).cpu().item() == timestep:
            break
        rand_mask_prob = cosine_schedule(timestep)
        num_token_masked = (rand_mask_prob * lens_t).long().clip(min=1)

        if token_cond is not None:
            num_token_masked = (rand_mask_prob * num_token_cond).long().clip(min=1)
            scores[token_cond != special_ids_t['mask_id']] = 0

        # Set sampling score to 0 for [PAD] and [CLS]
        scores[~seq_mask_no_end_t] = 0
        scores[:, 0] = 0
        scores = scores / scores.sum(-1)[:, None]  # normalize only unmasked token

        _, sorted_score_indices = scores.sort(descending=True)  # deterministic

        token_ids_t[~seq_mask_t] = special_ids_t['pad_id']  # replace with pad id
        token_ids_t.scatter_(-1, lens_t[..., None].long(), special_ids_t['eos_id'])  # replace with end id

        # replace "mask_id" to "ids" that have highest "num_token_masked" "scores"
        select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked)
        # repeat last_id to make it scatter_ the existing last ids
        last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1) - 1)
        sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index * ~select_masked_indices)
        token_ids_t.scatter_(-1, sorted_score_indices, special_ids_t['mask_id'])

        logits = model(token_ids_t, token_ids_m, seq_mask_t, seq_mask_m)

        if rand_pos:
            temperature = 1  # starting_temperature * (steps_until_x0 / timesteps)  # temperature is annealed
        else:
            temperature = 0  # starting_temperature * (steps_until_x0 / timesteps)  # temperature is annealed

        # if temperature == 0, it is equal to argmax (pred_ids = pred_t.argmax(dim=-1))
        pred_ids_t = gumbel_sample(logits['logits_t'], temperature=temperature, dim=-1)
        is_mask = token_ids_t == special_ids_t['mask_id']

        token_ids_t = torch.where(is_mask, pred_ids_t, token_ids_t)

        # Update score
        probs_without_temperature = logits['logits_t'].softmax(dim=-1)
        scores = 1 - probs_without_temperature.gather(-1, pred_ids_t[..., None])
        scores = rearrange(scores, '... 1 -> ...')
        scores = scores.masked_fill(~is_mask, 0)

    return token_ids_t

@torch.no_grad()
def eval_bitm_m2t(out_dir, val_loader, bitm, logger, writer, nb_iter, tokenizer, special_ids_t, invalid_ids, max_m, max_t,
                  best_iter=0., best_bleu1=0., best_bleu4=0., best_rouge_l=0., best_cider=0., best_bert_f1=0.,
                  draw=True, save=True, num_repeat=1, rand_pos=False):
    if num_repeat < 0:  # evaluate all generations
        is_avg_all = True
        num_repeat = -num_repeat
    else:  # evaluate last generation
        is_avg_all = False

    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # set up evaluators for CIDEr and BERT F1 if the libraries are available
    if NLGMetricverse is not None and load_metric is not None:
        # Only the CIDEr metric is loaded here; BLEU and ROUGE are computed elsewhere
        _cider_metrics = [load_metric("cider")]
        nlg_evaluator = NLGMetricverse(_cider_metrics)
    else:
        nlg_evaluator = None

    bitm.eval()
    nb_sample = 0
    bleu1 = 0.
    bleu4 = 0.
    rouge_l = 0.
    cider_score = 0.
    bert_f1 = 0.
    metric_batches = []

    for batch in tqdm(val_loader, position=2, leave=True):
        word_embeddings, pos_one_hots, sent_len, token_ids_m, pose, m_length, token_ids_t, seq_mask_t, captions = batch
        bs, seq = pose.shape[:2]
        lens_m = torch.ceil(m_length / 4).long()

        # Get lengths for each text in batch
        t_valid_mask = ~torch.isin(token_ids_t.cuda(), torch.tensor(invalid_ids).cuda())
        lens_t = t_valid_mask.sum(dim=1)

        # Get motion mask
        seq_mask_m = generate_src_mask(max_m, lens_m + 1)

        # generate target token masks
        seq_mask_t = generate_src_mask(max_t, lens_t + 1)  # target token: text
        seq_mask_no_end_t = generate_src_mask(max_t, lens_t)  # target token: text

        for i in range(num_repeat):
            index_text = inference_m2t(bitm,
                                       lens_t=lens_t.cuda(),
                                       token_ids_m=token_ids_m.cuda(),
                                       seq_mask_m=seq_mask_m.cuda(),
                                       seq_mask_t=seq_mask_t.cuda(),
                                       seq_mask_no_end_t=seq_mask_no_end_t.cuda(),
                                       special_ids_t=special_ids_t,
                                       max_length=max_t - 1,
                                       shape=(bs, max_t),
                                       rand_pos=rand_pos)  # (bs, max_t)
            pred_text = decode_token_ids(index_text, tokenizer, eos_id=special_ids_t['eos_id'])

            if i == 0 or is_avg_all:
                metric_batches.append((pred_text, captions, bs))
                nb_sample += bs

    for pred_text, captions, bs in metric_batches:
        rouge_l += compute_rouge_l(pred_text, captions, scorer=rouge_scorer_obj)
        bleu_scores = compute_bleu_scores(pred_text, captions)
        bleu1 += bleu_scores['BLEU-1']
        bleu4 += bleu_scores['BLEU-4']

        # compute CIDEr using nlgmetricverse (if available)
        if nlg_evaluator is not None:
            references = [[c] for c in captions]
            # nlg_evaluator returns a dict keyed by metric names
            _scores = nlg_evaluator(predictions=pred_text, references=references)
            cider_value = _scores["cider"]["score"]
            cider_score += cider_value
        # compute BERT F1 using bert_score (if available)
        if bert_score is not None:
            pred_text = list(pred_text)
            captions = list(captions)

            P, R, F1 = bert_score(pred_text, captions, lang="en", rescale_with_baseline=True, idf=True, verbose=False)
            bert_f1 += F1.sum().item()

    bleu1 = bleu1 / nb_sample
    bleu4 = bleu4 / nb_sample
    rouge_l = rouge_l / nb_sample

    cider_score = cider_score / nb_sample if nb_sample > 0 else 0.0
    bert_f1 = bert_f1 / nb_sample if nb_sample > 0 else 0.0

    msg = f"--> \t Eva. Iter {nb_iter} :, \n\
                bleu1. {bleu1}, \n\
                bleu4. {bleu4}, \n\
                rouge_l. {rouge_l:.4f}, \n\
                cidEr. {cider_score:.4f}, \n\
                bert_f1. {bert_f1:.4f}"

    logger.info(msg)

    if draw:
        writer.add_scalar('./Test/bleu1', bleu1, nb_iter)
        writer.add_scalar('./Test/bleu4', bleu4, nb_iter)
        writer.add_scalar('./Test/rouge_l', rouge_l, nb_iter)
        writer.add_scalar('./Test/cider', cider_score, nb_iter)
        writer.add_scalar('./Test/bert_f1', bert_f1, nb_iter)

    if bleu1 > best_bleu1:
        msg = f"--> --> \t BLEU1 Improved from {best_bleu1:.4f} to {bleu1:.4f} !!!"
        logger.info(msg)
        best_bleu1 = bleu1

    if bleu4 > best_bleu4:
        msg = f"--> --> \t BLEU4 Improved from {best_bleu4:.4f} to {bleu4:.4f} !!!"
        logger.info(msg)
        best_bleu4, best_iter = bleu4, nb_iter

    if rouge_l > best_rouge_l:
        msg = f"--> --> \t ROUGE-L Improved from {best_rouge_l:.4f} to {rouge_l:.4f} !!!"
        logger.info(msg)
        best_rouge_l = rouge_l

    if cider_score > best_cider:
        msg = f"--> -->\t CIDEr Improved from {best_cider:.4f} to {cider_score:.4f} !!!"
        logger.info(msg)
        best_cider = cider_score

    if bert_f1 > best_bert_f1:
        msg = f"--> -->\t BERT-F1 Improved from {best_bert_f1:.4f} to {bert_f1:.4f} !!!"
        logger.info(msg)
        best_bert_f1 = bert_f1

    if save:
        torch.save({'bitm': get_model(bitm).state_dict()}, os.path.join(out_dir, 'net_last_t.pth'))

    bitm.train()
    return best_iter, best_bleu1, best_bleu4, best_rouge_l, best_cider, best_bert_f1