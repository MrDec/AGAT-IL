from __future__ import division
from __future__ import print_function
import os, sys
import warnings

from sklearn.manifold import TSNE
from torch.nn import Parameter
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import Planetoid

from visiuation import plot_embedding

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# For replicating the experiments
SEED = 42
import argparse
import time
import random
import numpy as np
import scipy.sparse as sp
import torch, gc


np.random.seed(SEED)
torch.manual_seed(SEED)
from torch import optim, nn
import torch.nn.functional as F
from linearmodel import LinTrans, LogReg
from optimizer import loss_function
from utils import *
from sklearn.cluster import SpectralClustering, KMeans
from clustering_metric import clustering_metrics
from tqdm import tqdm
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from GATEModel import *

parser = argparse.ArgumentParser()
parser.add_argument('--linlayers', type=int, default=1, help="Number of hidden layers")  # 隐藏层层数
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')  # 多少个epoch
parser.add_argument('--dims', type=int, default=[64], help='Number of units in hidden layer 1.')  # 隐藏层的维度
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')  # 学习率
parser.add_argument('--upth_st', type=float, default=0.0110, help='Upper Threshold start.')  # 上边界的start
parser.add_argument('--lowth_st', type=float, default=0.1, help='Lower Threshold start.')  # 下边界的start
parser.add_argument('--upth_ed', type=float, default=0.0010, help='Upper Threshold end.')  # 上边界的end
parser.add_argument('--lowth_ed', type=float, default=0.5, help='Lower Threshold end.')  # 下边界的end
parser.add_argument('--upd', type=int, default=10, help='Update epoch.')  # 10个epoch更新一次
parser.add_argument('--bs', type=int, default=10000, help='Batchsize.')
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda is True:
    print('Using GPU')
    torch.cuda.manual_seed(SEED)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def clustering(Cluster, feature, true_labels):
    f_adj = np.matmul(feature, np.transpose(feature))  # matmul()函数是Numpy中的矩阵乘法函数 transpose 转换坐标轴 相当于求X*XT
    predict_labels = Cluster.fit_predict(f_adj)

    cm = clustering_metrics(true_labels, predict_labels)
    db = -metrics.davies_bouldin_score(f_adj, predict_labels)  # DBI 指数，越小越好，无监督分类的性能指标 类似的还有轮廓系数
    acc, nmi, adj, f1 = cm.evaluationClusterModelFromLabel(tqdm)  # 聚类评价指标

    return db, acc, nmi, adj, f1


# 更新相似性矩阵S
def update_similarity(z, upper_threshold, lower_treshold,pos_num, neg_num):
    f_adj = np.matmul(z, np.transpose(z))
    cosine = f_adj
    cosine = cosine.reshape([-1, ])
    pos_num = round(upper_threshold * len(cosine))  # round函数用于四舍五入
    neg_num = round((1 - lower_treshold) * len(cosine))

    pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
    neg_inds = np.argpartition(cosine, neg_num)[:neg_num]

    return np.array(pos_inds), np.array(neg_inds)  # 返回正样本  负样本下标s


# 更新边界
def update_threshold(upper_threshold, lower_treshold, up_eta, low_eta):
    upth = upper_threshold + up_eta
    lowth = lower_treshold + low_eta
    return upth, lowth

class STAGAM(nn.Module):

    def __init__(self, num_features,middle_size,representation_size, clusters_number, alpha,dims,v=1):
        super().__init__()
        self.v = v
        self.gate =  GATE(num_features, middle_size, representation_size, alpha)
        self.linear = LinTrans(layers=1,dims=dims)
        # cluster layer   # cluster layer，簇头embed
        self.cluster_layer = Parameter(torch.Tensor(clusters_number, dims[1]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, adj, M, xind, yind):
            A_pred, z_embedding = self.gate(x, adj, M)
            x = torch.index_select(z_embedding, 0, xind)
            y = torch.index_select(z_embedding, 0, yind)
            # outx, prex = self.linear(x)
            # outy, prey = self.linear(y)
            out, pre = self.linear(z_embedding)
            q = self.modularity(out)
            #return A_pred, z_embedding, q, outx, outy, out, pre
            return A_pred, z_embedding, q, out, pre

    def modularity(self, z_embedding):
        dist = torch.sum(torch.pow(z_embedding.unsqueeze(1) - self.cluster_layer, 2), 2)
        q = 1.0 / (1.0 + dist / self.v) ** ((self.v + 1.0) / 2.0)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    # t_order
    t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    # M就是论文中的proximity matrix M
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)

def target_fenbu(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def Modula(array, cluster):
    m = sum(sum(array)) / 2
    k1 = np.sum(array, axis=1)
    k2 = k1.reshape(k1.shape[0], 1)
    k1k2 = k1 * k2
    Eij = k1k2 / (2 * m)
    B = array - Eij
    node_cluster = np.dot(cluster, np.transpose(cluster))
    results = np.dot(B, node_cluster)
    sum_results = np.trace(results)
    modul = sum_results / (2 * m)
    return modul

def gae_for(args):
    print("Using {} dataset".format(args.dataset))
    if args.dataset == 'cora':
        n_clusters = 7
        Cluster = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
    elif args.dataset == 'citeseer':
        n_clusters = 6
        Cluster = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
    elif args.dataset == 'pubmed':
        n_clusters = 3
        Cluster = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
    elif args.dataset == 'wiki':
        n_clusters = 17
        Cluster = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)


    adj, features, true_labels, idx_train, idx_val, idx_test = load_data(args.dataset)  # 邻接矩阵 真实标签 训练集 验证集 测试集
    y = true_labels
    n_nodes, feat_dim = features.shape  # 结点数量 特征维度
    dims = [512] + args.dims  # args.dims 编码器隐藏层维度

    layers = args.linlayers  # 隐藏层数 1层
    # Store original adjacency matrix (without diagonal entries) for later
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)

    adj.eliminate_zeros()

    n = adj.shape[0]
    # gnnlayers = getbestlayer(adj,features,n_clusters)
    # print('gnnlayer:{}'.format(gnnlayers))
    # adj_norm_s = preprocess_graph(adj, gnnlayers, norm='sym', renorm=True)
    adj_norm_s = preprocess_graph(adj, 2, norm='sym', renorm=True)
    sm_fea_s = sp.csr_matrix(features).toarray()

    z_embeddin = TSNE(n_components=2).fit_transform(sm_fea_s)
    '''分隔符'''  # 这里是聚类可视化
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(z_embeddin[:, 0], z_embeddin[:, 1], c=y, s=20)
    color_set = ('gray', 'green', 'red', 'cyan', 'magenta', 'yellow', 'blue')
    color_list = [color_set[int(label)] for label in y]
    plt.scatter(z_embeddin[:, 0], z_embeddin[:, 1], c=color_list, s=30)
    plt.savefig("raw.png")
    print('Laplacian Smoothing...')

    for a in adj_norm_s:
        sm_fea_s = a.dot(sm_fea_s)  # 通过循环之后得到过滤后的特征 adj_norm_s里存放的是三个I-KL


    adj_1st = (adj + sp.eye(n)).toarray()
    adj_label = torch.FloatTensor(adj_1st)

    model = STAGAM(feat_dim, middle_size=1024, representation_size=512, clusters_number=n_clusters, alpha=0.2, dims=dims)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 使用Adam优化器

    #scheduler = StepLR(optimizer, step_size=200, gamma=0.5)

    # writer = SummaryWriter('logs')

    sm_fea_s = torch.FloatTensor(sm_fea_s)  # 类型转

    """经过过滤器"""
    z_embeddin = TSNE(n_components=2).fit_transform(sm_fea_s.data.cpu().numpy())
    '''分隔符'''  # 这里是聚类可视化
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(z_embeddin[:, 0], z_embeddin[:, 1], c=y, s=20)
    color_set = ('gray', 'green', 'red', 'cyan', 'magenta', 'yellow', 'blue')
    color_list = [color_set[int(label)] for label in y]
    plt.scatter(z_embeddin[:, 0], z_embeddin[:, 1], c=color_list, s=30)
    plt.savefig("filter.png")
    # fig = plot_embedding(inx_plot, y, '')
    # fig.show()

    true_labels = torch.LongTensor(true_labels)


    if args.cuda:
        model.cuda()
        inx = sm_fea_s.cuda()
        adj_label = adj_label.cuda()

    listacc = []
    listnmi = []
    listari = []
    listf1 = []

    ###########################
    pos_num = len(adj.indices)
    neg_num = n_nodes * n_nodes - pos_num
    up_eta = (args.upth_ed - args.upth_st) / (args.epochs / args.upd)
    low_eta = (args.lowth_ed - args.lowth_st) / (args.epochs / args.upd)
    pos_inds, neg_inds = update_similarity(normalize(sm_fea_s.numpy()), args.upth_st, args.lowth_st, pos_num, neg_num)
    upth, lowth = update_threshold(args.upth_st, args.lowth_st, up_eta, low_eta)
    bs = min(args.bs, len(pos_inds))
    pos_inds_cuda = torch.LongTensor(pos_inds).cuda()

    #################################
    adj = torch.FloatTensor(adj.toarray()).cuda()
    M = get_M(adj)
    print('Start Training...')

    for epoch in tqdm(range(args.epochs)):
        gc.collect()
        torch.cuda.empty_cache()
        st, ed = 0, bs
        batch_num = 0
        length = len(pos_inds)
        model.train()
        while (ed <= length):
            t = time.time()
            sampled_neg = torch.LongTensor(np.random.choice(neg_inds, size=ed - st)).cuda()
            sampled_inds = torch.cat((pos_inds_cuda[st:ed], sampled_neg), 0)
            optimizer.zero_grad()
            xind = sampled_inds // n_nodes
            yind = sampled_inds % n_nodes
            A_pred, z_embedding, Q, out, pre = model(inx, adj, M.cuda(),xind.cuda(),yind.cuda())

            ######################
            # A_pred, z_embedding, Q, outx, outy,out, pre = model(inx, adj, M.cuda(), xind.cuda(), yind.cuda())
            # batch_label = torch.cat((torch.ones(ed - st), torch.zeros(ed - st))).cuda()
            # batch_pred = model.linear.dcs(outx, outy)
            # t_loss = loss_function(adj_preds=batch_pred, adj_labels=batch_label)
            #######################

            ll_loss  = loss_function(adj_preds=pre,adj_labels=adj_label)
            p = target_fenbu(Q.detach())
            kl_loss = F.kl_div(Q.log(), p, reduction='batchmean')  # 自监督损失
            MU_loss = Modula(adj.detach().cpu().numpy(), pre.detach().cpu().numpy())  # Lm 模块度损失

            loss = ll_loss + 0.001 * kl_loss + 0.001 * MU_loss

            cur_loss = loss.item()
            loss.backward()
            optimizer.step()
            #scheduler.step()

            st = ed
            batch_num += 1
            if ed < length and ed + bs >= length:
                ed += length - ed
            else:
                ed += bs


        if (epoch + 1) % args.upd == 0:
            model.eval()
            with torch.no_grad():
                A_pred, z_embedding, q, mu, pre = model(inx, adj, M.cuda(),xind.cuda(),yind.cuda())
                """经过Attention"""
                z_embeddin = TSNE(n_components=2).fit_transform(z_embedding.data.cpu().numpy())
                '''分隔符'''  # 这里是聚类可视化
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.scatter(z_embeddin[:, 0], z_embeddin[:, 1], c=y, s=20)
                color_set = ('gray', 'green', 'red', 'cyan', 'magenta', 'yellow', 'blue')
                color_list = [color_set[int(label)] for label in y]
                plt.scatter(z_embeddin[:, 0], z_embeddin[:, 1], c=color_list, s=30)
                plt.savefig("attention/attention+{}.png".format(epoch+1))
                """经过adptive"""
                z_embeddin = TSNE(n_components=2).fit_transform(mu.data.cpu().numpy())
                '''分隔符'''  # 这里是聚类可视化
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.scatter(z_embeddin[:, 0], z_embeddin[:, 1], c=y, s=20)
                color_set = ('gray', 'green', 'red', 'cyan', 'magenta', 'yellow', 'blue')
                color_list = [color_set[int(label)] for label in y]
                plt.scatter(z_embeddin[:, 0], z_embeddin[:, 1], c=color_list, s=30)
                plt.savefig("adaptive/adaptive+{}.png".format(epoch+1))
                hidden_emb = mu.cpu().data.numpy()
                ########################
                upth, lowth = update_threshold(upth, lowth, up_eta, low_eta)
                pos_inds, neg_inds = update_similarity(hidden_emb, upth, lowth, pos_num, neg_num)
                bs = min(args.bs, len(pos_inds))
                pos_inds_cuda = torch.LongTensor(pos_inds).cuda()
                ###############################
                db, acc, nmi, adjscore, f1 = clustering(Cluster, hidden_emb, true_labels.cpu().data.numpy())
                tqdm.write("Epoch: {}, train_loss_gae={:.5f}, time={:.5f}, acc={:.5f}, nmi={:.5f}, adj={:.5f}, f1={:.5f}".format(
                    epoch + 1, cur_loss, time.time() - t, acc, nmi, adjscore, f1))
                # writer.add_scalar('loss', cur_loss, global_step= epoch)
                # writer.add_scalar('acc', acc, global_step=epoch)
                listacc.append(acc)
                listari.append(adjscore)
                listnmi.append(nmi)
                listf1.append(f1)
    tqdm.write("Optimization Finished!")
    tqdm.write('best_acc: {}, best_nmi: {}, best_adj: {}, best_f1: {}'.format(max(listacc), max(listnmi), max(listari),max(listf1)))
    # writer.close()

if __name__ == '__main__':
    gae_for(args)