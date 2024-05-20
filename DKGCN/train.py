import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score
from torch import optim
import time
from tensorboardX import SummaryWriter
from model import KGCN
import torch.nn.functional as F

class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_id = np.array(self.df[idx][0])
        item_id = np.array(self.df[idx][1])
        label = np.array(self.df[idx][2])
        return user_id, item_id, label

def dataload(args,data):
    n_user, n_item, kg = data[0], data[1], data[2]
    train_data, eval_data, test_data = data[3], data[4], data[5]
    n_entity, n_relation = data[6], data[7]

    train_dataset = KGCNDataset(train_data)
    eval_dataset = KGCNDataset(eval_data)
    test_dataset = KGCNDataset(test_data)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    return train_dataloader,eval_dataloader,test_dataloader,n_user, n_item, kg,n_entity, n_relation

def train(args, data):
    '''
    :param args: 参数设置
    :param data: 数据集
    :return:
    '''
    is2tensorboard = True
    names = args.sample
    writer = SummaryWriter("runs2/logs{}".format(names))
    name = str(args.n_iter)
    # 读取数据并且进行预处理
    train_dataloader,eval_dataloader,test_dataloader,n_user, n_item, kg ,n_entity, n_relation= dataload(args,data)
    # prepare network, loss function, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = KGCN(n_user, n_entity, n_relation, kg, args, device).to(device)
    print('n_user, n_entity, n_relation',n_user, n_entity, n_relation)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)  # 学习率每30个epoch衰减成原来的1/5

    print('device: ', device)

    # train
    auc = 60
    for epoch in range(args.n_epochs):
        b=time.time()
        running_loss = 0.0
        train_total_roc = 0
        train_total_f1 = 0
        X_list = []
        Y_list = torch.ones(n_entity).to(device)
        # for k in range(K):
        #     Y_list.append(torch.ones(n_entity).to(device))
        K = args.sample
        for k in range(K):
            X_list.append(F.dropout(torch.ones(n_entity), p=args.dropnode_rate).to(device))
    
        for i, (user_ids, item_ids, labels) in enumerate(train_dataloader):
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            output_list =[]
            loss_train = 0.
            for k in range(K):
                output_list.append(net(user_ids, item_ids,X_list[k]))
            # outputs = net(user_ids, item_ids)
            labels = labels.float()
            for k in range(K):
                loss_train += criterion(output_list[k], labels)  
            loss = loss_train/K
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_total_roc += roc_auc_score(labels.cpu().detach().numpy(), output_list[0].cpu().detach().numpy())
            # train_total_f1 += f1_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())

        # print train loss,auc,f1 per every epoch
        train_auc = train_total_roc / len(train_dataloader)
        train_loss = running_loss / len(train_dataloader)

        # ------------------------------------------------------------------
        scheduler.step()
        # train_f1 = train_total_f1 / len(train_dataloader)
        print('[Epoch {}]train_loss:{:.4} '.format(epoch + 1, train_loss), end="\t")
        print('train_auc: {:.4}'.format(train_auc), end="\t")
        write2tensorboard(is2tensorboard, "train" + name, writer, train_loss, train_auc, epoch)
        # print('train_f1: ', train_f1, end="\t")
        # evaluate per every epoch
        with torch.no_grad():
            test_loss = 0
            test_total_roc = 0
            test_total_f1 = 0
            for user_ids, item_ids, labels in test_dataloader:
                user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                
                outputs = net(user_ids, item_ids,Y_list)
                labels = labels.float()
                test_loss += criterion(outputs, labels).item()
                test_total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            #                test_total_f1 += f1_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            test_loss = test_loss / len(test_dataloader)
            test_auc = test_total_roc / len(test_dataloader)
            # test_f1 = test_total_f1 / len(test_dataloader)
            print('test_loss:{:.4} '.format(test_loss), end="\t")
            print('test_auc: {:.4}'.format(test_auc),end="\t")
            write2tensorboard(is2tensorboard, "test" + name, writer, test_loss, test_auc, epoch)
            # print('test_f1: ', test_f1)
            # if test_auc >= auc:
            #     torch.save(net,"net.pth")
            #     auc = test_auc    

        net.eval()
        eval_total_roc = 0
        eval_total_f1 = 0
        for _, (user_ids, item_ids, labels) in enumerate(eval_dataloader):
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            outputs = net(user_ids, item_ids,Y_list)
            labels = labels.float()
            eval_total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            outs = return_b(outputs.cpu().detach().numpy()) 
            eval_total_f1 += f1_score(labels.cpu().detach().numpy(), outs)

        eval_roc = eval_total_roc / len(eval_dataloader)
        eval_f1 = eval_total_f1 / len(eval_dataloader)    
        print('eval_roc:{:.4}'.format(eval_roc),end="\t")
        print('eval_f1:{:.4}'.format(eval_f1))
        e=time.time()
        print("time is ",e-b)


def write2tensorboard(is2tensorboard, name, writer, loss, auc, epoch):
    if is2tensorboard:
        writer.add_scalar(name + '_loss', loss, global_step=epoch)
        writer.add_scalar(name + '_auc', auc, global_step=epoch)
        # writer.add_scalar(name + '_f1', f1, global_step=epoch)


def return_b(outs):
    for i in range(len(outs)):
        if outs[i] >= 0.5:
            outs[i] = 1
        else:
            outs[i] = 0
    return outs
