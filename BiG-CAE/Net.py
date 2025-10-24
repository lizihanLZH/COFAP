import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import mean_nodes
from dgl.nn.pytorch import HeteroGraphConv, GraphConv

EMBEDDINGS = {
    "l2n": (300, 200),  # linker到node的嵌入维度
    "n2l": (200, 300),  # node到linker的嵌入维度
}

class GraphConvModel(nn.Module):
    def __init__(
        self,
        num_conv_layers=3,
        num_fc_layers=1,
        conv_dim=128,
        fc_dim=128,
        conv_norm="both",
        conv_activation=F.elu,
        fc_activation=F.relu,
        norm=nn.LayerNorm,
    ):
        super(GraphConvModel, self).__init__()
        self.num_fc_layers = num_fc_layers
        self.conv_activation = conv_activation
        self.fc_activation = fc_activation
        
        self.norms = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        
        # 第一层卷积
        self.conv_layers.append(
            HeteroGraphConv(
                {
                    k: GraphConv(v[0], conv_dim, norm=conv_norm)
                    for k, v in EMBEDDINGS.items()
                },
                aggregate="mean",
            )
        )
        self.norms.append(nn.ModuleDict({k: norm(conv_dim) for k in ("n", "l")}))
        
        # 后续卷积层
        for i in range(1, num_conv_layers):
            self.conv_layers.append(
                HeteroGraphConv(
                    {
                        k: GraphConv(conv_dim, conv_dim, norm=conv_norm)
                        for k, v in EMBEDDINGS.items()
                    },
                    aggregate="mean",
                )
            )
            self.norms.append(nn.ModuleDict({k: norm(conv_dim) for k in ("n", "l")}))
        
        # 全连接层
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(conv_dim, fc_dim))
        for l in range(1, num_fc_layers):
            self.fc_layers.append(nn.Linear(fc_dim, fc_dim))
        
        self.out = nn.Linear(fc_dim, 1)
    
    def forward(self, g):
        with g.local_scope():
            # 处理空图的情况
            if all(g.num_nodes(ntype) == 0 for ntype in g.ntypes):
               return torch.zeros(1,1, device=next(model.parameters()).device)

            
            h = {ntype: g.nodes[ntype].data["feat"] for ntype in g.ntypes}
            
            # 图卷积层
            for i in range(len(self.conv_layers)):
                h = self.conv_layers[i](g, (h, h))
                h = {k: self.norms[i][k](v) for k, v in h.items()}
                h = {k: self.conv_activation(v) for k, v in h.items()}
            
            g.ndata["h"] = h
            
            # 图级别的池化
            hg = 0
            for ntype in g.ntypes:
                if g.num_nodes(ntype) > 0:  # 确保节点类型不为空
                    hg = hg + mean_nodes(g, "h", ntype=ntype)
            
            # 全连接层
            for k in range(self.num_fc_layers):
                hg = self.fc_layers[k](hg)
                hg = self.fc_activation(hg)
            
            return self.out(hg)

def train_step(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    total_loss = 0.0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(y)
            print(f"train loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    
    return total_loss / num_batches

def valid_step(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
    
    total_loss /= num_batches
    print(f"valid loss: {total_loss:>8f} \n")
    return total_loss

def predict_dataloader(model, dataloader, device):
    model.eval()
    y_true = torch.tensor([], dtype=torch.float32, device=device)
    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_true = torch.cat((y_true, y), 0)
            y_pred = torch.cat((y_pred, model(X)), 0)
    
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return y_true, y_pred
