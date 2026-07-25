from models.HGT import *
from models.Seqmodels import *
from layers.TSEncoder import *
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from chatgpt_desc.load_embedding import *
from networks.recurent import LiquidRecurrent

feats_to_nodes = {
    'conds_hist': 'co',
    'procedures': 'pr',
    'drugs': 'dh',
    'co': 'conds_hist',
    'pr': 'procedures',
    'dh': 'drugs'
}

graph_meta = (['visit', 'co', 'pr', 'dh'],
 [('co', 'in', 'visit'),
  ('pr', 'in', 'visit'),
  ('dh', 'in', 'visit'),
  ('visit', 'connect', 'visit'),
  ('visit', 'has', 'co'),
  ('visit', 'has', 'pr'),
  ('visit', 'has', 'dh')])


class TRANS(nn.Module):
    def __init__(
            self,
            Tokenizers,
            hidden_size,
            output_size,
            device,
            graph_meta,
            embedding_dim=128,
            dropout=0.5,
            num_heads=2,
            num_layers=2,
            pe=False,
    ):
        super(TRANS, self).__init__()
        self.embedding_dim = embedding_dim
        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.feature_keys = Tokenizers.keys()
        self.device = device
        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)

        self.transformer = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.transformer[feature_key] = TransformerLayer(
                feature_size=embedding_dim, dropout=dropout
            )
        self.tim2vec = Time2Vec(8).to(device)
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)
        self.graphmodel = HGT(hidden_channels=hidden_size, out_channels=output_size, num_heads=num_heads,
                              num_layers=num_layers, metadata=graph_meta).to(device)
        self.pe = pe
        self.spatialencoder = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.spatialencoder[feature_key] = nn.Linear(self.pe * 2, embedding_dim)  # .to(self.device)
        self.alpha = 0.8

    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
            tokenizer.get_vocabulary_size(),
            self.embedding_dim,
            padding_idx=tokenizer.get_padding_index(),
        )

    def get_embedder(self):
        feature = {}
        for k in self.embeddings.keys():
            lenth = self.feat_tokenizers[k].get_vocabulary_size()
            tensor = torch.arange(0, lenth, dtype=torch.long).to(self.device)
            feature[k] = self.embeddings[k](tensor)
        return feature

    def process_seq(self, seqdata):
        patient_emb = []
        for feature_key in self.feature_keys:
            x = self.feat_tokenizers[feature_key].batch_encode_3d(
                seqdata[feature_key],
            )
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            x = self.embeddings[feature_key](x)
            x = torch.sum(x, dim=2)
            mask = torch.any(x != 0, dim=2)
            _, x = self.transformer[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        return logits, patient_emb

    def process_graph_fea(self, graph_list, pe):
        f = self.get_embedder()
        for i in range(len(graph_list)):
            for node_type, x in graph_list[i].x_dict.items():
                if node_type != 'visit':
                    if self.pe:
                        lpe = graph_list[i][node_type].laplacian_pe.to(self.device)
                        rws = graph_list[i][node_type].random_walk_se.to(self.device)
                        se = self.spatialencoder[feats_to_nodes[node_type]](torch.cat([lpe, rws], dim=-1))
                        # graph_list[i][node_type].x = torch.cat([f[feats_to_nodes[node_type]],\
                        #                                     lpe, \
                        #                                     rws], dim=-1)
                        graph_list[i][node_type].x = f[feats_to_nodes[node_type]] + se

                    else:
                        graph_list[i][node_type].x = f[feats_to_nodes[node_type]]
                if node_type == 'visit':
                    timevec = self.tim2vec(
                        torch.tensor(graph_list[i]['visit'].time, dtype=torch.float32, device=self.device))
                    num_visit = graph_list[i]['visit'].x.shape[0]
                    graph_list[i]['visit'].x = torch.cat([pe[i].repeat(num_visit, 1), timevec], dim=-1)
        return Batch.from_data_list(graph_list)

    def forward(self, batchdata):
        seq_logits, Patient_emb = self.process_seq(batchdata[0])
        graph_data = self.process_graph_fea(batchdata[1], Patient_emb).to(self.device)
        out = self.alpha * self.graphmodel(graph_data.edge_index_dict, graph_data) + (1 - self.alpha) * seq_logits
        return out

class HieraMedWL(nn.Module):
    def __init__(
        self,
        Tokenizers,
        hidden_size,
        output_size,
        device,
        graph_meta,
        unfolding_steps = 5,
        embedding_dim = 128,
        dropout = 0.5,
        num_heads = 2,
        num_layers = 2,
        pe = False,
    ):
        super(HieraMedWL, self).__init__()
        self.embedding_dim = embedding_dim
        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ParameterDict()
        self.linear_layers = nn.ModuleDict()
        self.feature_keys = Tokenizers.keys()
        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.unfolding_steps = unfolding_steps
        for feature_key in self.feature_keys:
            print(feature_key)
            self.add_feature_transform_layer(feature_key)

        self.transformer = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.transformer[feature_key] = TransformerLayer(
                feature_size=embedding_dim, dropout=dropout
            )

        self.delta_t = Parameter(torch.Tensor(60))
        self.prop = LiquidRecurrent(hidden_size, hidden_size, self.unfolding_steps, hidden_size).to(device)
        self.lin = nn.Linear(hidden_size, output_size)

        self.tim2vec = Time2Vec(8).to(device)
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)
        self.graphmodel = HGT(hidden_channels = hidden_size, out_channels = hidden_size, num_heads=num_heads, num_layers = num_layers, metadata = graph_meta).to(device)
        self.pe = pe
        self.spatialencoder = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.spatialencoder[feature_key] = nn.Linear(self.pe*2, embedding_dim)#.to(self.device)
        self.alpha = 0.8

    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
            tokenizer.get_vocabulary_size(),
            self.embedding_dim,
            padding_idx=tokenizer.get_padding_index(),
        )

    def get_embedder(self):
        feature = {}
        for k in self.embeddings.keys():
            lenth = self.feat_tokenizers[k].get_vocabulary_size()
            tensor = torch.arange(0, lenth, dtype=torch.long).to(self.device)
            feature[k] = self.embeddings[k](tensor)
        return feature

    def process_seq(self, seqdata):
        patient_emb = []
        for feature_key in self.feature_keys:
            x = self.feat_tokenizers[feature_key].batch_encode_3d(
                seqdata[feature_key],
            )
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            x = self.embeddings[feature_key](x)
            x = torch.sum(x, dim=2)
            mask = torch.any(x != 0, dim=2)
            _, x = self.transformer[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        return logits, patient_emb
    
    def process_graph_fea(self, graph_list, pe):
        f = self.get_embedder()
        for i in range(len(graph_list)):
            for j in range(len(graph_list[i])):
                for node_type, x in graph_list[i][j].x_dict.items():
                    if node_type!='visit':
                        if self.pe:
                            lpe = graph_list[i][j][node_type].laplacian_pe.to(self.device)
                            rws = graph_list[i][j][node_type].random_walk_se.to(self.device)
                            se = self.spatialencoder[feats_to_nodes[node_type]](torch.cat([lpe,rws], dim=-1))
                            # graph_list[i][node_type].x = torch.cat([f[feats_to_nodes[node_type]],\
                            #                                     lpe, \
                            #                                     rws], dim=-1)
                            graph_list[i][j][node_type].x = f[feats_to_nodes[node_type]] + se

                        else:
                            graph_list[i][j][node_type].x = f[feats_to_nodes[node_type]]
                    if node_type=='visit':
                        timevec = self.tim2vec(torch.tensor(graph_list[i][j]['visit'].time, dtype = torch.float32, device=self.device))
                        num_visit = graph_list[i][j]['visit'].x.shape[0]
                        #graph_list[i][j]['visit'].x = torch.cat([pe[i].repeat(num_visit, 1), timevec],dim=-1)
                        graph_list[i][j]['visit'].x = graph_list[i][j]['visit'].x
        return graph_list
    
    def forward1(self, batchdata):
        seq_logits, Patient_emb = self.process_seq(batchdata[0])
        graph_data = self.process_graph_fea(batchdata[1], Patient_emb).to(self.device)
        out = self.alpha * self.graphmodel(graph_data.edge_index_dict, graph_data) + (1-self.alpha) * seq_logits
        out = seq_logits
        return out

    def forward(self, batchdata):
        seq_logits, Patient_emb = self.process_seq(batchdata[0])
        graph_list = self.process_graph_fea(batchdata[1], Patient_emb)
        out_graph = torch.empty(len(graph_list), self.output_size).to(self.device)
        #print(len(graph_list))
        for i in range(len(graph_list)):
            #print(i)
            #print(len(graph_list[i]))
            graph_data = Batch.from_data_list(graph_list[i]).to(self.device)
            graph_embedding = self.graphmodel(graph_data.edge_index_dict, graph_data)
            out = self.prop(graph_embedding, self.delta_t)
            out_graph[i] = self.lin(out)

        out = self.alpha * out_graph + (1-self.alpha) * seq_logits
        #out = seq_logits
        return out

class MyEmbedding(nn.Module):
    def __init__(self,tokenizer, feature_key, embedding_dim, padding_idx=None):
        super().__init__()
        diag_embeddings, proce_embeddings, atc_embeddings = load_text_embeddings()
        diag_embeddings = reduce_embeddings(diag_embeddings, embedding_dim)
        proce_embeddings = reduce_embeddings(proce_embeddings, embedding_dim)
        atc_embeddings = reduce_embeddings(atc_embeddings, embedding_dim)
        vocab_emb = np.random.randn(tokenizer.get_vocabulary_size(), embedding_dim)
        list_ = list(range(tokenizer.get_vocabulary_size()))
        code_list = tokenizer.convert_indices_to_tokens(list_)
        if feature_key == 'condas_hist':
            for i in range(tokenizer.get_vocabulary_size()):
                if code_list[i] in diag_embeddings:
                    vocab_emb[i] = diag_embeddings[code_list[i]]
        if feature_key == 'procedures':
            for i in range(tokenizer.get_vocabulary_size()):
                if code_list[i] in proce_embeddings:
                    vocab_emb[i] = proce_embeddings[code_list[i]]
        if feature_key == 'drugs':
            for i in range(tokenizer.get_vocabulary_size()):
                if code_list[i] in atc_embeddings:
                    vocab_emb[i] = atc_embeddings[code_list[i]]
        vocab_emb = torch.tensor(vocab_emb, dtype=torch.float)
        self.embedding_weight = nn.Parameter(vocab_emb)
        #nn.init.xavier_uniform_(self.embedding_weight)
        nn.init.normal_(self.embedding_weight)

        self.padding_idx = padding_idx
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding_weight[self.padding_idx].fill_(0.0)  # 初始化为0

    def forward(self, input_ids):
        emb = F.embedding(input_ids, self.embedding_weight, padding_idx=self.padding_idx)
        return emb


class HieraMed(nn.Module):
    def __init__(
            self,
            Tokenizers,
            hidden_size,
            output_size,
            device,
            graph_meta,
            unfolding_steps=5,
            embedding_dim=128,
            dropout=0.5,
            num_heads=2,
            num_layers=2,
            pe=False,
    ):
        super(HieraMed, self).__init__()
        self.embedding_dim = embedding_dim
        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ParameterDict()
        self.linear_layers = nn.ModuleDict()
        self.feature_keys = Tokenizers.keys()
        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.unfolding_steps = unfolding_steps
        #self.lin_in = nn.Linear(768, 128)
        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)

        self.transformer = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.transformer[feature_key] = TransformerLayer(
                feature_size=embedding_dim, dropout=dropout
            )

        self.delta_t = Parameter(torch.Tensor(60))
        self.prop = LiquidRecurrent(hidden_size, hidden_size, self.unfolding_steps, hidden_size).to(device)
        self.lin = nn.Linear(hidden_size, output_size)

        self.tim2vec = Time2Vec(8).to(device)
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)
        self.graphmodel = HGT(hidden_channels=hidden_size, out_channels=hidden_size, num_heads=num_heads,
                              num_layers=num_layers, metadata=graph_meta).to(device)
        self.pe = pe
        self.spatialencoder = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.spatialencoder[feature_key] = nn.Linear(self.pe * 2, embedding_dim)  # .to(self.device)
        self.alpha = 0.8

    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        #nn.init.normal_(self.embeddings[feature_key], mean=0.0, std=1.0)
        #self.embeddings[feature_key] = self.lin_in(self.embeddings[feature_key])

        self.embeddings[feature_key] = MyEmbedding(
            tokenizer = tokenizer,
            feature_key = feature_key,
            embedding_dim = self.embedding_dim,
            padding_idx=tokenizer.get_padding_index()
        )

    def get_embedder(self):
        feature = {}
        for k in self.embeddings.keys():
            lenth = self.feat_tokenizers[k].get_vocabulary_size()
            tensor = torch.arange(0, lenth, dtype=torch.long).to(self.device)
            feature[k] = self.embeddings[k](tensor)
        return feature

    def process_seq(self, seqdata):
        patient_emb = []
        for feature_key in self.feature_keys:
            x = self.feat_tokenizers[feature_key].batch_encode_3d(
                seqdata[feature_key],
            )
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            x = self.embeddings[feature_key](x)
            x = torch.sum(x, dim=2)
            mask = torch.any(x != 0, dim=2)
            _, x = self.transformer[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        return logits, patient_emb

    def process_graph_fea(self, graph_list, pe):
        f = self.get_embedder()
        for i in range(len(graph_list)):
            for j in range(len(graph_list[i])):
                for node_type, x in graph_list[i][j].x_dict.items():
                    if node_type != 'visit':
                        if self.pe:
                            lpe = graph_list[i][j][node_type].laplacian_pe.to(self.device)
                            rws = graph_list[i][j][node_type].random_walk_se.to(self.device)
                            se = self.spatialencoder[feats_to_nodes[node_type]](torch.cat([lpe, rws], dim=-1))
                            # graph_list[i][node_type].x = torch.cat([f[feats_to_nodes[node_type]],\
                            #                                     lpe, \
                            #                                     rws], dim=-1)
                            graph_list[i][j][node_type].x = f[feats_to_nodes[node_type]] + se

                        else:
                            graph_list[i][j][node_type].x = f[feats_to_nodes[node_type]]
                    if node_type == 'visit':
                        timevec = self.tim2vec(
                            torch.tensor(graph_list[i][j]['visit'].time, dtype=torch.float32, device=self.device))
                        num_visit = graph_list[i][j]['visit'].x.shape[0]
                        # graph_list[i][j]['visit'].x = torch.cat([pe[i].repeat(num_visit, 1), timevec],dim=-1)
                        graph_list[i][j]['visit'].x = graph_list[i][j]['visit'].x
        return graph_list

    def forward1(self, batchdata):
        seq_logits, Patient_emb = self.process_seq(batchdata[0])
        graph_data = self.process_graph_fea(batchdata[1], Patient_emb).to(self.device)
        out = self.alpha * self.graphmodel(graph_data.edge_index_dict, graph_data) + (1 - self.alpha) * seq_logits
        out = seq_logits
        return out

    def forward(self, batchdata):
        #list_conds = list(range(self.feat_tokenizers['conds_hist'].get_vocabulary_size()))
        #conds_list = self.feat_tokenizers['conds_hist'].convert_indices_to_tokens(list_conds)
        #list_proc = list(range(self.feat_tokenizers['procedures'].get_vocabulary_size()))
        #proc_list = self.feat_tokenizers['procedures'].convert_indices_to_tokens(list_proc)
        #list_drugs = list(range(self.feat_tokenizers['drugs'].get_vocabulary_size()))
        #drugs_list = self.feat_tokenizers['drugs'].convert_indices_to_tokens(list_drugs)

        #return conds_list, proc_list, drugs_list, self.embeddings['conds_hist'].embedding_weight, self.embeddings['procedures'].embedding_weight, self.embeddings['drugs'].embedding_weight
        seq_logits, Patient_emb = self.process_seq(batchdata[0])
        graph_list = self.process_graph_fea(batchdata[1], Patient_emb)
        out_graph = torch.empty(len(graph_list), self.output_size).to(self.device)
        # print(len(graph_list))
        for i in range(len(graph_list)):
            # print(i)
            # print(len(graph_list[i]))
            graph_data = Batch.from_data_list(graph_list[i]).to(self.device)
            graph_embedding = self.graphmodel(graph_data.edge_index_dict, graph_data)
            out = self.prop(graph_embedding, self.delta_t)
            out_graph[i] = self.lin(out)

        out = self.alpha * out_graph + (1 - self.alpha) * seq_logits
        # out = seq_logits
        return out


'''
class HieraMed(nn.Module):
    def __init__(
            self,
            Tokenizers,
            hidden_size,
            output_size,
            device,
            graph_meta,
            unfolding_steps=5,
            embedding_dim=128,
            dropout=0.5,
            num_heads=2,
            num_layers=2,
            pe=False,
    ):
        super(HieraMed, self).__init__()
        self.embedding_dim = embedding_dim
        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ParameterDict()
        self.linear_layers = nn.ModuleDict()
        self.feature_keys = Tokenizers.keys()
        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.unfolding_steps = unfolding_steps
        #self.lin_in = nn.Linear(768, 128)
        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)

        self.transformer = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.transformer[feature_key] = TransformerLayer(
                feature_size=embedding_dim, dropout=dropout
            )

        self.delta_t = Parameter(torch.Tensor(60))
        self.prop = LiquidRecurrent(hidden_size, hidden_size, self.unfolding_steps, hidden_size).to(device)
        self.lin = nn.Linear(hidden_size, output_size)

        self.tim2vec = Time2Vec(8).to(device)
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)
        self.graphmodel = HGT(hidden_channels=hidden_size, out_channels=hidden_size, num_heads=num_heads,
                              num_layers=num_layers, metadata=graph_meta).to(device)
        self.pe = pe
        self.spatialencoder = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.spatialencoder[feature_key] = nn.Linear(self.pe * 2, embedding_dim)  # .to(self.device)
        self.alpha = 0.8

    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        #nn.init.normal_(self.embeddings[feature_key], mean=0.0, std=1.0)
        #self.embeddings[feature_key] = self.lin_in(self.embeddings[feature_key])

        self.embeddings[feature_key] = MyEmbedding(
            tokenizer = tokenizer,
            feature_key = feature_key,
            embedding_dim = self.embedding_dim,
            padding_idx=tokenizer.get_padding_index()
        )
        self.embeddings[feature_key] = nn.Embedding(
            tokenizer.get_vocabulary_size(),
            self.embedding_dim,
            padding_idx=tokenizer.get_padding_index()
        )

    def get_embedder(self):
        feature = {}
        for k in self.embeddings.keys():
            lenth = self.feat_tokenizers[k].get_vocabulary_size()
            tensor = torch.arange(0, lenth, dtype=torch.long).to(self.device)
            feature[k] = self.embeddings[k](tensor)
        return feature

    def process_seq(self, seqdata):
        patient_emb = []
        for feature_key in self.feature_keys:
            x = self.feat_tokenizers[feature_key].batch_encode_3d(
                seqdata[feature_key],
            )
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            x = self.embeddings[feature_key](x)
            x = torch.sum(x, dim=2)
            mask = torch.any(x != 0, dim=2)
            _, x = self.transformer[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        return logits, patient_emb

    def process_graph_fea(self, graph_list, pe):
        f = self.get_embedder()
        for i in range(len(graph_list)):
            for j in range(len(graph_list[i])):
                for node_type, x in graph_list[i][j].x_dict.items():
                    if node_type != 'visit':
                        if self.pe:
                            lpe = graph_list[i][j][node_type].laplacian_pe.to(self.device)
                            rws = graph_list[i][j][node_type].random_walk_se.to(self.device)
                            se = self.spatialencoder[feats_to_nodes[node_type]](torch.cat([lpe, rws], dim=-1))
                            # graph_list[i][node_type].x = torch.cat([f[feats_to_nodes[node_type]],\
                            #                                     lpe, \
                            #                                     rws], dim=-1)
                            graph_list[i][j][node_type].x = f[feats_to_nodes[node_type]] + se

                        else:
                            graph_list[i][j][node_type].x = f[feats_to_nodes[node_type]]
                    if node_type == 'visit':
                        timevec = self.tim2vec(
                            torch.tensor(graph_list[i][j]['visit'].time, dtype=torch.float32, device=self.device))
                        num_visit = graph_list[i][j]['visit'].x.shape[0]
                        # graph_list[i][j]['visit'].x = torch.cat([pe[i].repeat(num_visit, 1), timevec],dim=-1)
                        graph_list[i][j]['visit'].x = graph_list[i][j]['visit'].x
        return graph_list

    def forward(self, batchdata):
        #return self.embeddings['conds_hist'].embedding_weight, self.embeddings['procedures'].embedding_weight, self.embeddings['drugs'].embedding_weight
        seq_logits, Patient_emb = self.process_seq(batchdata[0])
        graph_list = self.process_graph_fea(batchdata[1], Patient_emb)
        out_graph = torch.empty(len(graph_list), self.output_size).to(self.device)
        # print(len(graph_list))
        for i in range(len(graph_list)):
            # print(i)
            # print(len(graph_list[i]))
            graph_data = Batch.from_data_list(graph_list[i]).to(self.device)
            graph_embedding = self.graphmodel(graph_data.edge_index_dict, graph_data)
            out = self.prop(graph_embedding, self.delta_t)
            out_graph[i] = self.lin(out)

        out = self.alpha * out_graph + (1 - self.alpha) * seq_logits
        # out = seq_logits
        return out
'''
