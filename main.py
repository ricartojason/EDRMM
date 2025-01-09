import os
import torch
import numpy as np
import dill
import pickle as pkl
import argparse
from torch.optim import Adam
from modules.MoleRec import MoleRecModel
from modules.gnn import graph_batch_from_smile
from util import buildPrjSmiles
from training import Test, Train


def set_seed(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model_name(args):
    model_name = [
        f'dim_{args.dim}',  f'lr_{args.lr}', f'coef_{args.coef}',
        f'dp_{args.dp}', f'ddi_{args.target_ddi}'
    ]
    if args.embedding:
        model_name.append('embedding')
    return '-'.join(model_name)


def parse_args():
    parser = argparse.ArgumentParser('Experiment For DrugRec')
    parser.add_argument('--Test', action='store_true', help="evaluating mode")
    parser.add_argument('--dim', default=128, type=int, help='model dimension')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--dp', default=0.8, type=float, help='dropout ratio')
    parser.add_argument(
        '--gnn_type', default='gin' , type=str, 
        help= 'choose the gnn_type from [gin,gcn,gat]'
    )
    parser.add_argument(
        '--model_name', type=str,
        help="the model name for training, if it's left blank,"
        " the code will generate a new model name on it own"
    )
    parser.add_argument(
        '--resume_path', type=str,
        help='path of well trained model, only for evaluating the model'
    )
    parser.add_argument(
        '--device', type=int, default=1,
        help='gpu id to run on, negative for cpu'
    )
    parser.add_argument(
        '--target_ddi', type=float, default=0.06,
        help='expected ddi for training'
    )
    parser.add_argument(
        '--coef', default=2.5, type=float,
        help='coefficient for DDI Loss Weight Annealing'
    )
    parser.add_argument(
        '--embedding', action='store_true',
        help='use embedding table for substructures' +
        'if it\'s not chosen, the substructure will be encoded by GNN'
    )
    parser.add_argument(
        '--epochs', default=100, type=int,
        help='the epochs for training'
    )
    parser.add_argument(
        '--seed', default=3025, type=int,
        help='seed for initializing training.'
    )
    parser.add_argument(
        '--deug', default=True, action='store_true', help='enable debug mode'
    )

    args = parser.parse_args()
    if args.Test and args.resume_path is None:
        raise FileNotFoundError('Can\'t Load Model Weight From Empty Dir')
    if args.model_name is None:
        args.model_name = get_model_name(args)


    return args


if __name__ == '__main__':
    
    args = parse_args()
    set_seed(args.seed)
    print(args)
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    
    data_with_sym_path = '/home/wwh/EDRMM/baseline/DrugRec/data/mimic-iii/output/records_ori_iii.pkl'
    
    voc_with_sym_path = '/home/wwh/EDRMM/baseline/DrugRec/data/mimic-iii/output/voc_iii_sym1_mulvisit.pkl'
    
    ddi_adj_path = '/home/wwh/EDRMM/baseline/DrugRec/data/mimic-iii/output/ddi_A_iii.pkl'
    ddi_adj_weighted_path = '/home/wwh/EDRMM/baseline/DrugRec/data/mimic-iii/output/ddi_A_weighted_iii.pkl'
    ehr_adj_path = '/home/wwh/EDRMM/baseline/DrugRec/data/mimic-iii/output/ehr_adj_iii.pkl'
    
    ehr_adj_weighted_path = '/home/wwh/EDRMM/baseline/DrugRec/data/mimic-iii/output/ehr_A_weighted_iii.pkl'
    
    ddi_mask_path = '/home/wwh/EDRMM/MHGRL/data/ddi_mask_H.pkl'
    
    molecule_path = '/home/wwh/EDRMM/MHGRL/data/idx2SMILES.pkl'
    substruct_smile_path = '/home/wwh/EDRMM/MHGRL/data/substructure_smiles.pkl'
    sym_embs = '/home/wwh/EDRMM/MHGRL/data/sym_embs.pkl'

    with open(sym_embs, 'rb') as Fin:
        sym_embs = dill.load(Fin)

    with open(data_with_sym_path, 'rb') as Fin:
        data = dill.load(Fin)

    with open(ddi_adj_path, 'rb') as Fin:
        ddi_adj = dill.load(Fin)
    with open(ddi_adj_weighted_path, 'rb') as Fin:
        ddi_adj_weighted = torch.from_numpy(dill.load(Fin)).to(device)
    with open(ehr_adj_path, 'rb') as Fin:
        ehr_adj = dill.load(Fin)
    with open(ehr_adj_weighted_path, 'rb') as Fin:
        ehr_adj_weighted = torch.from_numpy(dill.load(Fin)).to(device)

    # 分子和子结构之间的掩码矩阵，如果药物分子i包含子结构j，矩阵的第i行第j列设置为1
    with open(ddi_mask_path, 'rb') as Fin:
        ddi_mask_H = torch.from_numpy(dill.load(Fin)).to(device)

    '''with open(data_path, 'rb') as Fin:
        data = dill.load(Fin)'''
    with open(molecule_path, 'rb') as Fin:
        molecule = dill.load(Fin)

    # 诊断、药物、手术、症状词表
    with open(voc_with_sym_path, 'rb') as Fin:
        voc = dill.load(Fin)

    split_point = int(len(data) * 2 / 3)

    
    data_train = data[:split_point]
    # 1/6的长度
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]
  
    diag_voc, pro_voc, med_voc, sym_voc = \
        voc['diag_voc'], voc['pro_voc'], voc['med_voc'], voc['sym_voc']
    voc_size = (
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word),
        len(sym_voc.idx2word)
    )
    print('四个词表的长度', voc_size)

    from collections import defaultdict
    # 使用defaultdict自动初始化计数器,最后按就诊次数排序
    visit_num = defaultdict(int)
    for patient in data_test:
        for i in range(len(patient)):
            visit_num[i+1] += 1
    visit_num = {k: v for k, v in sorted(visit_num.items(), key=lambda item: item[0])}
    # 创建药物-smiles映射
    average_projection, smiles_list = \
        buildPrjSmiles(molecule, med_voc.idx2word)
    average_projection = average_projection.to(device)

    molecule_graphs = graph_batch_from_smile(smiles_list)
    # molecule_graphs存储了284个smiles对应的图信息
    molecule_forward = {'batched_data': molecule_graphs.to(device)}
    # 设置分子图神经网络的参数
    molecule_para = {
        'num_layer': 4, 'emb_dim': args.dim, 'graph_pooling': 'mean',
        'drop_ratio': args.dp, 'gnn_type': args.gnn_type, 'virtual_node': False
    }

    if args.embedding:
        substruct_para, substruct_forward = None, None
    else:
        with open(substruct_smile_path, 'rb') as Fin:
            substruct_smiles_list = dill.load(Fin)
        #substruct_graphs存储了492个substructure smiles对应的图信息
        substruct_graphs = graph_batch_from_smile(substruct_smiles_list)
        substruct_forward = {'batched_data': substruct_graphs.to(device)}
        substruct_para = {
            'num_layer': 4, 'emb_dim': args.dim, 'graph_pooling': 'mean',
            'drop_ratio': args.dp, 'gnn_type': args.gnn_type, 'virtual_node': False
        }
    # 处理药物-药物相互作用(DDI)邻接矩阵
    adj_matrix = ddi_adj_weighted.cpu()
    # 找到非零元素的坐标
    # torch.nonzero() 返回张量中非零元素的索引
    # as_tuple=True 时返回元组 as_tuple=False时返回张量
    # t()转置操作
    edge_index = torch.nonzero(adj_matrix, as_tuple=False).t()
    edge_weight = adj_matrix[edge_index[0], edge_index[1]]
    ddi_adj_dict = {'edge_index': edge_index, 'edge_weight': edge_weight}
    adj_matrix = ehr_adj_weighted.cpu()
    # 找到非零元素的坐标
    edge_index = torch.nonzero(adj_matrix, as_tuple=False).t()
    edge_weight = adj_matrix[edge_index[0], edge_index[1]]
    ehr_adj_dict = {'edge_index': edge_index, 'edge_weight': edge_weight}
    model = MoleRecModel(
        global_para=molecule_para, substruct_para=substruct_para,
        emb_dim=args.dim, global_dim=args.dim, substruct_dim=args.dim,
        substruct_num=ddi_mask_H.shape[1], voc_size=voc_size, vocab = voc,
        use_embedding=args.embedding, device=device, dropout=args.dp,
        ddi_adj=ddi_adj, ehr_adj=ehr_adj, sym_embs=sym_embs.squeeze(1).to(device),
    ).to(device)

    drug_data = {
        'substruct_data': substruct_forward,
        'mol_data': molecule_forward,
        'ddi_mask_H': ddi_mask_H,
        'tensor_ddi_adj': torch.from_numpy(ddi_adj).to(device),
        'tensor_ehr_adj': ehr_adj_weighted,
        'average_projection': average_projection
    }
    
    if args.Test:
        Test(model, args.resume_path, device, data_test, voc_size, drug_data)
    else:
        if not os.path.exists(os.path.join('/home/wwh/EDRMM/MHGRL/saved', args.model_name)):
            os.makedirs(os.path.join('/home/wwh/EDRMM/MHGRL/saved', args.model_name))
        log_dir = os.path.join('../saved', args.model_name)
        optimizer = Adam(model.parameters(), lr=args.lr)
        Train(
            model, device, data_train, data_eval,voc_size, drug_data,
            optimizer, args.coef, args.target_ddi, args.seed, EPOCH=args.epochs, Args = args
        )