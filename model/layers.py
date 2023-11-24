from os import device_encoding
from helper import *
from torch.nn import Parameter
from .message_passing import MessagePassing
import torch
from torch_scatter import scatter_add
import time
class DWGCN(MessagePassing):
	def __init__(self, in_channels, out_channels, num_rels, act=lambda x:x, params=None):
		super(self.__class__, self).__init__()

		self.p 			= params
		self.in_channels	= in_channels
		self.out_channels	= out_channels
		self.num_rels 		= num_rels
		self.act 		= act
		self.device		= None

		self.w_loop		= get_param((in_channels, out_channels))
		self.w_in		= get_param((in_channels, out_channels))
		self.w_out		= get_param((in_channels, out_channels))
		self.w_rel 		= get_param((in_channels, out_channels))
		self.loop_rel 		= get_param((1, in_channels))

		self.drop		= torch.nn.Dropout(self.p.dropout)
		self.bn			= torch.nn.BatchNorm1d(out_channels)

		if self.p.bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

	def forward(self, x, edge_index, edge_type, edge_weight, rel_embed): 
		if self.device is None:
			self.device = edge_index.device

		rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
		num_edges = edge_index.size(1) // 2
		num_ent   = x.size(0)

		self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
		self.in_type,  self.out_type  = edge_type[:num_edges], 	 edge_type [num_edges:]
		self.in_wei, self.out_wei = edge_weight[num_edges, :], edge_weight[num_edges, :]
		#print("self.in_wei.size: {}".format(self.in_wei.size()))

		self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
		self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)
		self.in_norm     = self.compute_norm(self.in_index,  num_ent)
		self.out_norm    = self.compute_norm(self.out_index, num_ent)
		
		st1 = time.time()
		in_res		= self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,  edge_wei=self.in_wei, rel_embed=rel_embed, edge_norm=self.in_norm, 	mode='in')
		st2 = time.time()
		loop_res	= self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, edge_wei=None, rel_embed=rel_embed, edge_norm=None, 		mode='loop')
		st3 = time.time()
		out_res		= self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,  edge_wei=self.out_wei, rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out')
		st4 = time.time()
		#out		= self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)

		out1 = self.drop(in_res)*(1/2) + self.drop(out_res)*(1/2)  ##邻居聚合的信息
		#out1 = in_res*(1/2) + out_res*(1/2)  ##邻居聚合的信息
		L2 = torch.norm(out1 - loop_res, dim=1) ## 聚合信息与自身信息的L2距离
		#temp1 = torch.ones_like(L2) - 1/2*L2  ##lamda=0.5  max(1-1/2*L2, 0) trad_factor_list   lamda = 0.4 
        # temp1 = torch.ones_like(L2) - a/(2*(1-a)*L2) ##a=0.4
		a=0.5     
		#temp1 = torch.ones_like(L2) - a/(2*(1-a)*L2) ##a=0.4
		temp1 =a/(a+L2)
		temp1 = torch.unsqueeze(temp1, 1)
		zeros = torch.zeros_like(temp1)
		trad_factor_list,_ = torch.cat((temp1,zeros),dim=1).max(dim=1)  #  返回平衡因子,衡量本身信息占多少
		out = torch.mul(torch.unsqueeze(trad_factor_list,dim=-1),loop_res) + (torch.mul(torch.unsqueeze((1-trad_factor_list),dim=-1),out1))
		#out = self.drop(out)
		if self.p.bias: out = out + self.bias
		out = self.bn(out)
		#print("in_Res:",st2 - st1)
		#print("loop_res:",st3-st2)
		#print("out_res:",st4-st3)
		return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]		# Ignoring the self loop inserted

	def rel_transform(self, ent_embed, rel_embed):
		if   self.p.opn == 'corr': 	trans_embed  = ccorr(ent_embed, rel_embed)
		elif self.p.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
		elif self.p.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
		else: raise NotImplementedError

		return trans_embed

	def message(self, x_j, edge_type, edge_wei, rel_embed, edge_norm, mode):
		#edge_wei 每条边的权重
		if edge_wei:
			weight 	= getattr(self, 'w_{}'.format(mode))
			rel_emb = torch.index_select(rel_embed, 0, edge_type)
			xj_rel  = self.rel_transform(x_j, rel_emb)
			xj_rel  = torch.mul(xj_rel,edge_wei)
			out	= torch.mm(xj_rel, weight)
		else:
			weight 	= getattr(self, 'w_{}'.format(mode))
			rel_emb = torch.index_select(rel_embed, 0, edge_type)
			xj_rel  = self.rel_transform(x_j, rel_emb)
			out	= torch.mm(xj_rel, weight)

		return out if edge_norm is None else out * edge_norm.view(-1, 1)

	def update(self, aggr_out):
		return aggr_out

	def compute_norm(self, edge_index, num_ent):
		row, col	= edge_index
		edge_weight 	= torch.ones_like(row).float()
		deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
		deg_inv		= deg.pow(-0.5)							# D^{-0.5}
		deg_inv[deg_inv	== float('inf')] = 0
		norm		= deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

		return norm

	def __repr__(self):
		return '{}({}, {}, num_rels={})'.format(
			self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
