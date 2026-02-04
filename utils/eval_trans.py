import os

import clip
import numpy as np
import torch
from scipy import linalg

# import visualization.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric
from exit.utils import get_model, generate_src_mask
from tqdm import tqdm

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


def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
    xyz = xyz[:1]
    bs, seq = xyz.shape[:2]
    xyz = xyz.reshape(bs, seq, -1, 3)
    plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(), title_batch, outname)
    plot_xyz = np.transpose(plot_xyz, (0, 1, 4, 2, 3))
    writer.add_video(tag, plot_xyz, nb_iter, fps=20)


@torch.no_grad()
def evaluation_vqvae(out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1,
                     best_top2, best_top3, best_matching, eval_wrapper, draw=True, save=True, savegif=False,
                     savenpy=False):
    net.eval()
    nb_sample = 0

    draw_org = []
    draw_pred = []
    draw_text = []

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 21 if motion.shape[-1] == 251 else 22

        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        for i in range(bs):
            pose = val_loader.dataset.inv_transform(motion[i:i + 1, :m_length[i], :].detach().cpu().numpy())
            # pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

            pred_pose, loss_commit, perplexity = net(motion[i:i + 1, :m_length[i]])
            # pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
            # pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

            # if savenpy:
            #     np.save(os.path.join(out_dir, name[i]+'_gt.npy'), pose_xyz[:, :m_length[i]].cpu().numpy())
            #     np.save(os.path.join(out_dir, name[i]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

            pred_pose_eval[i:i + 1, :m_length[i], :] = pred_pose

            # if i < min(4, bs):
            #     draw_org.append(pose_xyz)
            #     draw_pred.append(pred_xyz)
            #     draw_text.append(caption[i])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

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

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)

    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)

        # if nb_iter % 5000 == 0 :
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)

        # if nb_iter % 5000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)   

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        # if save:
        #     torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity
        # if save:
        #     torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]
        # if save:
        #     torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred
        # if save:
        #     torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))

    if save:
        torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger


@torch.no_grad()
def evaluation_transformer_old(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div,
                           best_top1, best_top2, best_top3, best_matching, clip_model, eval_wrapper, dataname='t2m',
                           draw=True, save=True, savegif=False, num_repeat=1, rand_pos=False, CFG=-1):
    if num_repeat < 0:
        is_avg_all = True
        num_repeat = -num_repeat
    else:
        is_avg_all = False

    trans.eval()
    nb_sample = 0

    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    blank_id = get_model(trans).num_vq
    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch

        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22

        text = clip.tokenize(clip_text, truncate=True).cuda()

        feat_clip_text, word_emb = clip_model(text)

        motion_multimodality_batch = []
        m_tokens_len = torch.ceil((m_length) / 4)

        pred_len = m_length.cuda()
        pred_tok_len = m_tokens_len

        for i in range(num_repeat):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            # pred_len = torch.ones(bs).long()

            index_motion = trans(feat_clip_text, word_emb, type="sample", m_length=pred_len, rand_pos=rand_pos, CFG=CFG)
            # [INFO] 1. this get the last index of blank_id
            # pred_length = (index_motion == blank_id).int().argmax(1).float()
            # [INFO] 2. this get the first index of blank_id
            pred_length = (index_motion >= blank_id).int()
            pred_length = torch.topk(pred_length, k=1, dim=1).indices.squeeze().float()
            # pred_length[pred_length==0] = index_motion.shape[1] # if blank_id in the first frame, set length to max
            # [INFO] need to run single sample at a time b/c it's conv
            for k in range(bs):
                ######### [INFO] Eval only the predicted length
                #     if pred_length[k] == 0:
                #         pred_len[k] = seq
                #         continue
                #     pred_pose = net(index_motion[k:k+1, :int(pred_length[k].item())], type='decode')
                #     cur_len = pred_pose.shape[1]

                #     pred_len[k] = min(cur_len, seq)
                #     pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
                # et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
                ######################################################

                ######### [INFO] Eval by m_length
                pred_pose = net(index_motion[k:k + 1, :int(pred_tok_len[k].item())], type='decode')
                pred_pose_eval[k:k + 1, :int(pred_len[k].item())] = pred_pose
            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                              m_length)
            ######################################################

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))

            if i == 0 or is_avg_all:
                pose = pose.cuda().float()

                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                # if draw:
                #     pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                #     pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                #     for j in range(min(4, bs)):
                #         draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                #         draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3,
                                                           sum_all=True)
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

        # if nb_iter % 10000 == 0 :
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
        # if nb_iter % 10000 == 0 :
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        # if save:
        #     torch.save({'trans' : get_model(trans).state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
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
        torch.save({'trans': get_model(trans).state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return pred_pose_eval, pose, m_length, clip_text, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, multimodality, writer, logger

@torch.no_grad()
def evaluation_transformer(out_dir, val_loader, net, trans, logger, writer, nb_iter,
                           best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching,
                           clip_model, eval_wrapper, dataname='t2m', draw=True, save=True, savegif=False, num_repeat=1,
                           rand_pos=False, CFG=-1):
    if num_repeat < 0:
        is_avg_all = True
        num_repeat = -num_repeat
    else:
        is_avg_all = False

    trans.eval()
    nb_sample = 0

    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    blank_id = get_model(trans).num_vq
    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch

        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22

        text = clip.tokenize(clip_text, truncate=True).cuda()

        feat_clip_text, word_emb = clip_model(text)

        motion_multimodality_batch = []
        m_tokens_len = torch.ceil((m_length) / 4)

        pred_len = m_length.cuda()
        pred_tok_len = m_tokens_len

        for i in range(num_repeat):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            # pred_len = torch.ones(bs).long()

            index_motion = trans(feat_clip_text, word_emb, type="sample", m_length=pred_len, rand_pos=rand_pos, CFG=CFG)
            # [INFO] 1. this get the last index of blank_id
            # pred_length = (index_motion == blank_id).int().argmax(1).float()
            # [INFO] 2. this get the first index of blank_id
            pred_length = (index_motion >= blank_id).int()
            pred_length = torch.topk(pred_length, k=1, dim=1).indices.squeeze().float()
            # pred_length[pred_length==0] = index_motion.shape[1] # if blank_id in the first frame, set length to max
            # [INFO] need to run single sample at a time b/c it's conv
            for k in range(bs):
                ######### [INFO] Eval only the predicted length
                #     if pred_length[k] == 0:
                #         pred_len[k] = seq
                #         continue
                #     pred_pose = net(index_motion[k:k+1, :int(pred_length[k].item())], type='decode')
                #     cur_len = pred_pose.shape[1]

                #     pred_len[k] = min(cur_len, seq)
                #     pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
                # et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
                ######################################################

                ######### [INFO] Eval by m_length
                pred_pose = net(index_motion[k:k + 1, :int(pred_tok_len[k].item())], type='decode')
                pred_pose_eval[k:k + 1, :int(pred_len[k].item())] = pred_pose
            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                              m_length)
            ######################################################

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))

            if i == 0 or is_avg_all:
                pose = pose.cuda().float()

                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                # if draw:
                #     pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                #     pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                #     for j in range(min(4, bs)):
                #         draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                #         draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3,
                                                           sum_all=True)
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

        # if nb_iter % 10000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
        # if nb_iter % 10000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        # if save:
        #     torch.save({'trans' : get_model(trans).state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
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
        torch.save({'trans': get_model(trans).state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return pred_pose_eval, pose, m_length, clip_text, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, multimodality, writer, logger


@torch.no_grad()
def eval_trans_m(out_dir, val_loader, net, trans, logger, writer, nb_iter,
                 eval_wrapper, max_m, max_t, first,
                 best_iter=0, best_fid=1000, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100,
                 draw=True, save=True, num_repeat=1, rand_pos=False):
    if num_repeat < 0:  # evaluate all generations
        is_avg_all = True
        num_repeat = -num_repeat
    else:  # evaluate last generation
        is_avg_all = False

    trans.eval()
    nb_sample = 0
    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    for batch in tqdm(val_loader, position=1, leave=True):
        word_embeddings, pos_one_hots, sent_len, _, pose, m_length, input_ids, input_attn_mask, _ = batch
        bs, seq = pose.shape[:2]
        lens_m = torch.ceil(m_length / 4).long()
        pred_len = m_length.cuda()

        word_emb = trans.module.bert(input_ids=input_ids.cuda(), attention_mask=input_attn_mask.cuda())

        motion_multimodality_batch = []

        for i in range(num_repeat):
            index_motion = trans(type='sample_m', lens_m=lens_m.cuda(), word_emb=word_emb, seq_mask_t=input_attn_mask,
                                 max_m=max_m, max_t=max_t, rand_pos=rand_pos, first=first)  # (bs, max_m)

            # [INFO] need to run single sample at a time because it's conv
            pred_pose_eval = torch.zeros(pose.shape).cuda()
            for k in range(bs):
                # [INFO] Eval by m_length
                pred_pose = net(index_motion[k:k + 1, :lens_m[k].item()], type='decode')  # (1, m_length, dim_m)
                pred_pose_eval[k:k + 1, :pred_len[k].item()] = pred_pose  # (bs, m_length, dim_m)

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                              m_length)

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))

            if i == 0 or is_avg_all:
                pose = pose.cuda().float()

                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match

                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3,
                                                           sum_all=True)
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
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
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
        torch.save({'trans': get_model(trans).state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return best_iter, best_fid, best_div, best_top1, best_top2, best_top3, best_matching, multimodality


@torch.no_grad()
def eval_trans_t(out_dir, val_loader, net, trans, logger, writer, nb_iter,
                 eval_wrapper, tokenizer, special_ids, invalid_ids, max_m, max_t, first,
                 best_iter=0., best_bleu1=0., best_bleu2=0., best_bleu3=0., best_bleu4=0., best_rouge_l=0.,
                 best_cider=0., best_bert_f1=0.,
                 draw=True, save=True, num_repeat=1, rand_pos=False):
    if num_repeat < 0:  # evaluate all generations
        is_avg_all = True
        num_repeat = -num_repeat
    else:  # evaluate last generation
        is_avg_all = False

    # set up evaluators for CIDEr and BERT F1 if the libraries are available
    if NLGMetricverse is not None and load_metric is not None:
        # Only the CIDEr metric is loaded here; BLEU and ROUGE are computed elsewhere
        _cider_metrics = [load_metric("cider")]
        nlg_evaluator = NLGMetricverse(_cider_metrics)
    else:
        nlg_evaluator = None

    # bert_score is a function imported above, it will be None if the package is missing

    trans.eval()
    nb_sample = 0

    bleu1, bleu2, bleu3, bleu4, rouge_l = 0., 0., 0., 0., 0.
    cider_score = 0.
    bert_f1 = 0.

    metric_batches = []

    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    count = 0
    for batch in tqdm(val_loader, position=2, leave=True):
        # if count == 2:
        #     continue
        # count += 1
        word_embeddings, pos_one_hots, sent_len, token_ids_m, pose, m_length, input_ids, input_attn_mask, captions = batch
        bs, seq = pose.shape[:2]
        lens_m = torch.ceil(m_length / 4).long()

        # Get lengths for each text in batch
        t_valid_mask = ~torch.isin(input_ids.cuda(), torch.tensor(invalid_ids).cuda())
        lens_t = t_valid_mask.sum(dim=1)

        # Get motion mask
        seq_mask_m = generate_src_mask(max_m, lens_m + 1)

        for i in range(num_repeat):
            index_text = trans(type='sample_t', special_ids=special_ids, lens_t=lens_t, token_ids_m=token_ids_m,
                               seq_mask_m=seq_mask_m,
                               max_m=max_m, max_t=max_t, rand_pos=rand_pos, first=first)  # (bs, max_m)

            pred_text = decode_token_ids(index_text, tokenizer, pad_token_id=special_ids['pad_id'],
                                         eos_token_id=special_ids['eos_id'])

            if i == 0 or is_avg_all:
                metric_batches.append((pred_text, captions, bs))
                nb_sample += bs

    for pred_text, captions, bs in metric_batches:
        rouge_l += compute_rouge_l(pred_text, captions, scorer=rouge_scorer_obj)
        bleu_scores = compute_bleu_scores(pred_text, captions)
        bleu1 += bleu_scores['BLEU-1']
        bleu2 += bleu_scores['BLEU-2']
        bleu3 += bleu_scores['BLEU-3']
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

            P, R, F1 = bert_score(pred_text, captions, lang="en",
                                  rescale_with_baseline=True, idf=True,
                                  verbose=False)
            bert_f1 += F1.mean().item()

    bleu1 = bleu1 / nb_sample
    bleu2 = bleu2 / nb_sample
    bleu3 = bleu3 / nb_sample
    bleu4 = bleu4 / nb_sample
    rouge_l = rouge_l / nb_sample

    cider_score = cider_score / nb_sample if nb_sample > 0 else 0.0
    bert_f1 = bert_f1 / nb_sample if nb_sample > 0 else 0.0

    msg = f"--> \t Eva. Iter {nb_iter} :, \n\
                bleu1. {bleu1}, \n\
                bleu2. {bleu2}, \n\
                bleu3. {bleu3}, \n\
                bleu4. {bleu4}, \n\
                rouge_l. {rouge_l:.4f}, \n\
                cidEr. {cider_score:.4f}, \n\
                bert_f1. {bert_f1:.4f}"

    logger.info(msg)

    if draw:
        writer.add_scalar('./Test/bleu1', bleu1, nb_iter)
        writer.add_scalar('./Test/bleu2', bleu2, nb_iter)
        writer.add_scalar('./Test/bleu3', bleu3, nb_iter)
        writer.add_scalar('./Test/bleu4', bleu4, nb_iter)
        writer.add_scalar('./Test/rouge_l', rouge_l, nb_iter)
        writer.add_scalar('./Test/cider', cider_score, nb_iter)
        writer.add_scalar('./Test/bert_f1', bert_f1, nb_iter)

    if bleu1 > best_bleu1:
        msg = f"--> --> \t BLEU1 Improved from {best_bleu1:.4f} to {bleu1:.4f} !!!"
        logger.info(msg)
        best_bleu1 = bleu1

    if bleu2 > best_bleu2:
        msg = f"--> --> \t BLEU2 Improved from {best_bleu2:.4f} to {bleu2:.4f} !!!"
        logger.info(msg)
        best_bleu2 = bleu2

    if bleu3 > best_bleu3:
        msg = f"--> --> \t BLEU3 Improved from {best_bleu3:.4f} to {bleu3:.4f} !!!"
        logger.info(msg)
        best_bleu3 = bleu3

    if bleu4 > best_bleu4:
        msg = f"--> --> \t BLEU4 Improved from {best_bleu4:.4f} to {bleu4:.4f} !!!"
        logger.info(msg)
        best_bleu4, best_iter = bleu4, nb_iter

    if rouge_l > best_rouge_l:
        msg = f"--> --> \t ROUGE_L Improved from {best_rouge_l:.4f} to {rouge_l:.4f} !!!"
        logger.info(msg)
        best_rouge_l = rouge_l

    if cider_score > best_cider:
        msg = f"--> -->\tCIDEr Improved from {best_cider:.4f} to {cider_score:.4f} !!!"
        logger.info(msg)
        best_cider = cider_score

    if bert_f1 > best_bert_f1:
        msg = f"--> -->\tBERT_F1 Improved from {best_bert_f1:.4f} to {bert_f1:.4f} !!!"
        logger.info(msg)
        best_bert_f1 = bert_f1

    if save:
        torch.save({'trans': get_model(trans).state_dict()}, os.path.join(out_dir, 'net_last_t.pth'))

    trans.train()
    # return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, multimodality
    return best_iter, best_bleu1, best_bleu2, best_bleu3, best_bleu4, best_rouge_l, best_cider, best_bert_f1


def evaluation_transformer_uplow(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter,
                                 best_div, best_top1, best_top2, best_top3, best_matching, clip_model, eval_wrapper,
                                 dataname, draw=True, save=True, savegif=False, num_repeat=1, rand_pos=False, CFG=-1):
    from utils.humanml_utils import HML_UPPER_BODY_MASK, HML_LOWER_BODY_MASK

    trans.eval()
    nb_sample = 0

    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    blank_id = get_model(trans).num_vq
    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        pose = pose.cuda().float()
        pose_lower = pose[..., HML_LOWER_BODY_MASK]
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22

        text = clip.tokenize(clip_text, truncate=True).cuda()

        feat_clip_text, word_emb = clip_model(text)

        motion_multimodality_batch = []
        m_tokens_len = torch.ceil((m_length) / 4)

        pred_len = m_length.cuda()
        pred_tok_len = m_tokens_len

        max_motion_length = int(seq / 4) + 1
        mot_end_idx = get_model(net).vqvae.num_code
        mot_pad_idx = get_model(net).vqvae.num_code + 1
        target_lower = []
        for k in range(bs):
            target = net(pose[k:k + 1, :m_length[k]], type='encode')
            if m_tokens_len[k] + 1 < max_motion_length:
                target = torch.cat([target,
                                    torch.ones((1, 1, 2), dtype=int, device=target.device) * mot_end_idx,
                                    torch.ones((1, max_motion_length - 1 - m_tokens_len[k].int().item(), 2), dtype=int,
                                               device=target.device) * mot_pad_idx], axis=1)
            else:
                target = torch.cat([target,
                                    torch.ones((1, 1, 2), dtype=int, device=target.device) * mot_end_idx], axis=1)
            target_lower.append(target[..., 1])
        target_lower = torch.cat(target_lower, axis=0)

        for i in range(num_repeat):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            # pred_len = torch.ones(bs).long()

            index_motion = trans(feat_clip_text, target_lower, word_emb, type="sample", m_length=pred_len,
                                 rand_pos=rand_pos, CFG=CFG)
            # [INFO] 1. this get the last index of blank_id
            # pred_length = (index_motion == blank_id).int().argmax(1).float()
            # [INFO] 2. this get the first index of blank_id
            pred_length = (index_motion >= blank_id).int()
            pred_length = torch.topk(pred_length, k=1, dim=1).indices.squeeze().float()
            # pred_length[pred_length==0] = index_motion.shape[1] # if blank_id in the first frame, set length to max
            # [INFO] need to run single sample at a time b/c it's conv
            for k in range(bs):
                ######### [INFO] Eval only the predicted length
                #     if pred_length[k] == 0:
                #         pred_len[k] = seq
                #         continue
                #     pred_pose = net(index_motion[k:k+1, :int(pred_length[k].item())], type='decode')
                #     cur_len = pred_pose.shape[1]

                #     pred_len[k] = min(cur_len, seq)
                #     pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
                # et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
                ######################################################

                ######### [INFO] Eval by m_length
                all_tokens = torch.cat([
                    index_motion[k:k + 1, :int(pred_tok_len[k].item()), None],
                    target_lower[k:k + 1, :int(pred_tok_len[k].item()), None]
                ], axis=-1)
                pred_pose = net(all_tokens, type='decode')
                pred_pose_eval[k:k + 1, :int(pred_len[k].item())] = pred_pose
            pred_pose_eval[..., HML_LOWER_BODY_MASK] = pose_lower
            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                              m_length)
            ######################################################

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))

            if i == 0:
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                # if draw:
                #     pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                #     pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                #     for j in range(min(4, bs)):
                #         draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                #         draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3,
                                                           sum_all=True)
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

        # if nb_iter % 10000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
        # if nb_iter % 10000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        # if save:
        #     torch.save({'trans' : get_model(trans).state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
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
        torch.save({'trans': get_model(trans).state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return pred_pose_eval, pose, m_length, clip_text, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, multimodality, writer, logger


@torch.no_grad()
def evaluation_transformer_test(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div,
                                best_top1, best_top2, best_top3, best_matching, best_multi, clip_model, eval_wrapper,
                                draw=True, save=True, savegif=False, savenpy=False):
    trans.eval()
    nb_sample = 0

    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0

    for batch in val_loader:

        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22

        text = clip.tokenize(clip_text, truncate=True).cuda()

        feat_clip_text = clip_model.encode_text(text).float()
        motion_multimodality_batch = []
        for i in range(30):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()

            for k in range(bs):
                try:
                    index_motion = trans.sample(feat_clip_text[k:k + 1], True)
                except:
                    index_motion = torch.ones(1, 1).cuda().long()

                pred_pose = net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k + 1, :cur_len] = pred_pose[:, :seq]

                if i == 0 and (draw or savenpy):
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if savenpy:
                        np.save(os.path.join(out_dir, name[k] + '_pred.npy'), pred_xyz.detach().cpu().numpy())

                    if draw:
                        if i == 0:
                            draw_pred.append(pred_xyz)
                            draw_text_pred.append(clip_text[k])
                            draw_name.append(name[k])

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                              pred_len)

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))

            if i == 0:
                pose = pose.cuda().float()

                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw or savenpy:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                    if savenpy:
                        for j in range(bs):
                            np.save(os.path.join(out_dir, name[j] + '_gt.npy'),
                                    pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())

                    if draw:
                        for j in range(bs):
                            draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                            draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3,
                                                           sum_all=True)
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
    multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    logger.info(msg)

    if draw:
        for ii in range(len(draw_org)):
            tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/' + draw_name[ii] + '_org', nb_vis=1,
                                      title_batch=[draw_text[ii]], outname=[
                    os.path.join(out_dir, draw_name[ii] + '_skel_gt.gif')] if savegif else None)

            tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/' + draw_name[ii] + '_pred', nb_vis=1,
                                      title_batch=[draw_text_pred[ii]], outname=[
                    os.path.join(out_dir, draw_name[ii] + '_skel_pred.gif')] if savegif else None)

    trans.train()
    return fid, best_iter, diversity, R_precision[0], R_precision[1], R_precision[
        2], matching_score_pred, multimodality, writer, logger


# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
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
        #         print(correct_vec, bool_mat[:, i])
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


def decode_token_ids(token_ids, tokenizer, pad_token_id, eos_token_id):
    decoded_texts = []

    for ids in token_ids:
        ids = ids.tolist()

        clean_ids = []
        for t in ids:
            if t == eos_token_id:
                break
            if t == pad_token_id:
                continue
            clean_ids.append(t)

        text = tokenizer.decode(clean_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_texts.append(text.strip())

    return decoded_texts


def compute_bleu_scores(preds, refs):
    bleu1 = bleu2 = bleu3 = bleu4 = 0.0

    for pred, ref in zip(preds, refs):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        bleu1 += sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0))
        bleu2 += sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0))
        bleu3 += sentence_bleu([ref_tokens], pred_tokens, weights=(1 / 3, 1 / 3, 1 / 3, 0))
        bleu4 += sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))

    return {
        "BLEU-1": bleu1 / len(preds),
        "BLEU-2": bleu2 / len(preds),
        "BLEU-3": bleu3 / len(preds),
        "BLEU-4": bleu4 / len(preds),
    }


def compute_rouge_l(preds, refs, scorer=None):
    if scorer is None:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    rouge_l_f = 0.0

    for pred, ref in zip(preds, refs):
        score = scorer.score(ref, pred)
        rouge_l_f += score['rougeL'].fmeasure

    return rouge_l_f / len(preds)


def evaluate_text_generation(index_text, gt_text, tokenizer, pad_token_id, eos_token_id):
    pred_text = decode_token_ids(index_text, tokenizer, pad_token_id, eos_token_id)
    bleu_scores = compute_bleu_scores(pred_text, gt_text)
    rouge_l = compute_rouge_l(pred_text, gt_text)

    return {
        **bleu_scores,
        'ROUGE-L': rouge_l
    }
