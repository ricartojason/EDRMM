import os
import torch
import numpy as np
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, ehr_rate_score
import time
from collections import defaultdict
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import multilabel_margin_loss
import math
import dill
import logging
from datetime import datetime
import csv

def setup_training_logger(saved_dir, args):
    """设置训练日志"""
    # 记录超参数配置
    logger.info("=== Experiment Configuration ===")
    logger.info("Hyperparameters:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    logger.info("============================\n")
    
    return logger


def get_time_string():
    now = datetime.now()
    current_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
    return current_datetime

model_name = 'MoleRec'
saved_dir = os.path.join("saved", model_name, get_time_string())

os.makedirs(saved_dir, exist_ok=True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(
    saved_dir, 'log.txt'), encoding='utf8')
formatter = logging.Formatter(
    '%(asctime)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def eval_one_epoch(model, data_eval, voc_size, drug_data):
    with open('/home/wwh/EDRMM/baseline/DrugRec/data/mimic-iii/output/voc_iii_sym1_mulvisit.pkl', 'rb') as Fin:
        voc = dill.load(Fin)
    #读取csv做为字典
    # Read the procedure dictionary
    import pandas as pd

    # Read the procedure dictionary
    proc_dict = {}
    proc_data = pd.read_excel('/home/liufeiyan/T630-BACKUP/MHGRL/data/input/ICD9_PROCEDURE.xlsx')
    for index, row in proc_data.iterrows():
        proc_dict[row['PROCEDURE CODE']] = row['SHORT DESCRIPTION']

    # Read the diagnosis dictionary
    diag_dict = {}
    diag_data = pd.read_excel('/home/liufeiyan/T630-BACKUP/MHGRL/data/input/ICD9_DIAGNOSIS.xlsx')
    for index, row in diag_data.iterrows():
        diag_dict[row['DIAGNOSIS CODE']] = row['SHORT DESCRIPTION']

    med_voc=voc['med_voc']
    diag_voc = voc['diag_voc']
    proc_voc = voc['pro_voc']
    sym_voc = voc['sym_voc']
    model = model.eval()
    smm_record, ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(6)]
    med_cnt, visit_cnt = 0, 0
    ja_by_visit = [[] for _ in range(5)]
    auc_by_visit = [[] for _ in range(5)]
    pre_by_visit = [[] for _ in range(5)]
    recall_by_visit = [[] for _ in range(5)]
    f1_by_visit = [[] for _ in range(5)]
    smm_record_by_visit = [[] for _ in range(5)]
    case_study = []
    attention_len = []
    for step, input_seq in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input_seq):
           
            output, *_ ,len1,len2,len3,idx1,idx2,idx3= model(
                patient_data=input_seq[:adm_idx + 1],
                **drug_data
            )
            
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            output = torch.sigmoid(output).detach().cpu().numpy()[0]
            y_pred_prob.append(output)

            y_pred_tmp = output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)
            if adm_idx>0:
                attention_len.append([step,adm_idx,len1,len2,len3,jaccard_similarity(adm[2],y_pred_label_tmp)])
            if step==78 and adm_idx==4:
                print(len1,len2,len3)
                for v in input_seq[:adm_idx+1]:
                    print('diagnosis')
                    id = [diag_voc.idx2word.get(k) for k in v[0]]
                    name = [diag_dict.get(k) for k in id]
                    print(id)
                    print(name)

                    print('procedure')
                    id = [proc_voc.idx2word.get(k) for k in v[1]]
                    name = [proc_dict.get(int(k)) for k in id]
                    print(id)
                    print(name)

                    print('medication')
                    id = [med_voc.idx2word.get(k) for k in v[2]]
                    print(id)
             
                    print('symptom')
                    id = [sym_voc.idx2word.get(k) for k in v[3]]
                    print(id)
                print('predited')
                print([med_voc.idx2word.get(k) for k in y_pred_label_tmp])
                print(idx1,idx2,idx3)
                
                exit()
        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(step + 1, len(data_eval)))
        # 计算y_gt 和smm_record的重合元素的个数
        for j, visit in enumerate(y_gt):
            temp = np.where(visit == 1)[0]
            temp = sorted(temp)
            correct = set(temp) & set(y_pred_label[j])
            wrong = set(y_pred_label[j]) - set(temp)
            case_study.append([step,temp,y_pred_label[j],correct,wrong, len(correct), len(wrong)])
        for i in range(min(len(y_gt), 5)):
            # single_ja, single_p, single_r, single_f1 = sequence_metric_v2(np.array(y_gt[i:i+1]), np.array(y_pred[i:i+1]), np.array(y_pred_label[i:i+1]))
            single_ja, single_auc, single_p, single_r, single_f1 = multi_label_metric(np.array([y_gt[i]]), np.array([y_pred[i]]), np.array([y_pred_prob[i]]))
            ja_by_visit[i].append(single_ja)
            auc_by_visit[i].append(single_auc)
            pre_by_visit[i].append(single_p)
            recall_by_visit[i].append(single_r)
            f1_by_visit[i].append(single_f1)
            smm_record_by_visit[i].append(y_pred_label[i:i+1])
            
    ddi_rate = ddi_rate_score(smm_record)
    output_str = '\nDDI Rate: {:.4f}, Jaccard: {:.4f}, PRAUC: {:.4f}, ' +\
        'AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}\n'
    logger.info(output_str.format(
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p),
        np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))
    print('\tvisit1\tvisit2\tvisit3\tvisit4\tvisit5')
    print('count:', [len(buf) for buf in ja_by_visit])
    print('jaccard:', [np.mean(buf) for buf in ja_by_visit])
    print('auc:', [np.mean(buf) for buf in auc_by_visit])
    print('precision:', [np.mean(buf) for buf in pre_by_visit])
    print('recall:', [np.mean(buf) for buf in recall_by_visit])
    print('f1:', [np.mean(buf) for buf in f1_by_visit])
    print('DDI:', [ddi_rate_score(buf) for buf in smm_record_by_visit])
    
    
    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), \
        np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt

def Test(model, model_path, device, data_test, voc_size, drug_data):
    with open(model_path, 'rb') as Fin:
        model.load_state_dict(torch.load(Fin, map_location=device))
    model = model.to(device).eval()
    print('--------------------Begin Testing--------------------')
    ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
    tic, result, sample_size = time.time(), [], round(len(data_test) * 1)
    np.random.seed(0)
    for _ in range(1):
        '''test_sample = np.random.choice(data_test, sample_size, replace=True)'''
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = \
            eval_one_epoch(model, data_test, voc_size, drug_data)
        result.append([ddi_rate, ja, avg_f1, prauc, avg_med])
    result = np.array(result)
    mean, std = result.mean(axis=0), result.std(axis=0)
    metric_list = ['ddi_rate', 'ja', 'avg_f1', 'prauc', 'med']
    outstring = ''.join([
        "{}:\t{:.4f} $\\pm$ {:.4f} & \n".format(metric_list[idx], m, s)
        for idx, (m, s) in enumerate(zip(mean, std))
    ])
    print(outstring)
    print('average test time: {}'.format((time.time() - tic) / 10))
    print('parameters', get_n_params(model))

from torch.utils.tensorboard import SummaryWriter
def Train(
    model, device, data_train, data_eval,voc_size, drug_data,
    optimizer,  coef, target_ddi, seed,EPOCH, Args
):
    # 设置日志记录器
    logger = setup_training_logger(saved_dir, Args)

    history, best_epoch, best_ja = defaultdict(list), 0, 0
    total_train_time, ddi_losses, ddi_values = 0, [], []
    # 每k个step记一次loss
    k = 100
    writer = SummaryWriter(log_dir = 'logs_main')
    batch = 0
    batch_loss = 0
    total_step = 1
    for epoch in range(EPOCH):
        logger.info(f'----------------Epoch {epoch + 1}------------------')
        logger.info(f'with sym, seed{seed}, coef{coef}, with rnn, with histrory, with attribute-level similar extractor,with same extractor, with code-level attention')
        model = model.train()
        tic, ddi_losses_epoch = time.time(), []

        
        for step, input_seq in enumerate(data_train):
            for adm_idx, adm in enumerate(input_seq):
                '''bce_target = torch.zeros((1, voc_size[2])).to(device)
                bce_target[:, seq_data_train[step][adm_idx][2]] = 1

                multi_target = -torch.ones((1, voc_size[2])).long()
                for idx, item in enumerate(seq_data_train[step][adm_idx][2]):
                    multi_target[0][idx] = item
                multi_target = multi_target.to(device)
                input = input_seq[:adm_idx]
                input.append(train_subgraphs[step][adm_idx])
                result, loss_ddi = model(
                    patient_data=input,
                    **drug_data
                )'''
                bce_target = torch.zeros((1, voc_size[2])).to(device)
                bce_target[:, adm[2]] = 1

                multi_target = -torch.ones((1, voc_size[2])).long()
                for idx, item in enumerate(adm[2]):
                    multi_target[0][idx] = item
                multi_target = multi_target.to(device)

                result, loss_ddi, ddi_risk, *_ = model(
                    patient_data=input_seq[:adm_idx+1],
                    **drug_data
                )

                sigmoid_res = torch.sigmoid(result)

                loss_bce = binary_cross_entropy_with_logits(result, bce_target)
                loss_multi = multilabel_margin_loss(sigmoid_res, multi_target)

                result = sigmoid_res.detach().cpu().numpy()[0]
                result[result >= 0.5] = 1
                result[result < 0.5] = 0
                y_label = np.where(result == 1)[0]
                current_ddi_rate = ddi_rate_score(
                    [[y_label]]
                )
                    
                if current_ddi_rate > target_ddi:

                    beta = coef * (1 - (current_ddi_rate / target_ddi))
                
                    beta = min(math.exp(beta), 1)

                    gamma = 0.5
                    # combined with ddi risk term
                    loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) \
                        + (1 - beta) * (gamma * loss_ddi + (1-gamma) * ddi_risk)
                else:
                    loss = 0.95* loss_bce + 0.05* loss_multi

                batch_loss += loss

                ddi_losses_epoch.append(loss_ddi.detach().cpu().item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_step += 1
                if total_step % k == 0:
                    writer.add_scalar('loss',batch_loss / k, batch)
                    batch_loss = 0
                    batch += 1

            llprint('\rtraining step: {} / {}'.format(step, len(data_train)))
        ddi_losses.append(sum(ddi_losses_epoch) / len(ddi_losses_epoch))
        print(f'\nddi_loss : {ddi_losses[-1]}\n')
        train_time, tic = time.time() - tic, time.time()
        total_train_time += train_time
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = \
            eval_one_epoch(model, data_eval,voc_size, drug_data)
        print(f'training time: {train_time}, testing time: {time.time() - tic}')
        ddi_values.append(ddi_rate)
        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        if epoch >= 5:
            print('ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}'.format(
                np.mean(history['ddi_rate'][-5:]),
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['prauc'][-5:])
            ))

        model_name = 'Epoch_{}_TARGET_{:.2f}_JA_{:.4f}_DDI_{:.4f}.model'.format(
            epoch, target_ddi, ja, ddi_rate
        )
        torch.save(model.state_dict(), os.path.join(saved_dir, model_name))
        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja
        print('best_epoch: {}'.format(best_epoch))
        with open(os.path.join(saved_dir, 'best.txt'), 'a') as Fout:
            Fout.write(f'{best_epoch}\n')
        with open(os.path.join(saved_dir, 'ddi_losses.txt'), 'w') as Fout:
            for dloss, dvalue in zip(ddi_losses, ddi_values):
                Fout.write(f'{dloss}\t{dvalue}\n')

        with open(os.path.join(saved_dir, 'history.pkl'), 'wb') as Fout:
            dill.dump(history, Fout)
    print('avg training time/epoch: {:.4f}'.format(total_train_time / EPOCH))
