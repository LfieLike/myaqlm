""" Core mathematics for Additive Quantization (AQ): initialization, reconstruction and beam search"""
import random
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm.auto import trange

from src.kmeans import find_nearest_cluster, fit_faiss_kmeans, fit_kmeans, fit_kmeans_1d
from src.utils import ellipsis, maybe_script


class QuantizedLinear(nn.Module):
    def __init__(self, quantized_weight, bias: Optional[nn.Parameter]):
        super().__init__()
        self.out_features, self.in_features = quantized_weight.out_features, quantized_weight.in_features
        self.quantized_weight = quantized_weight
        self.bias = bias
        self.use_checkpoint = False

    def _forward(self, input: torch.Tensor):
        return F.linear(input, self.quantized_weight(), self.bias)

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint and torch.is_grad_enabled():
            return checkpoint(
                self._forward, input, use_reentrant=False, preserve_rng_state=False, determinism_check="none"
            )
        return self._forward(input)

def quantize(org_weight,codebook_num = 2,centroids_num = 256,block_size = 64,centroid_len = 8):
    # 计算每一行的二范数
    # max_matrix = get_max(org_weight)
    reshspe_weight = org_weight.view(-1,block_size)
    scales = reshspe_weight.norm(p=2, dim=1, keepdim=True).float()
    # nn.Parameter(scales, requires_grad=True)
    # 每一行除以其对应的范数
    normalized_tensor = (reshspe_weight / scales)
    weight_list = normalized_tensor.split(normalized_tensor.shape[0]//codebook_num,dim = 0)
    clusters_list = []
    nearest_indices_list = []
    # reconstructed_data_list = []
    for weight in weight_list:
        clusters, nearest_indices, _=fit_kmeans(weight.view(-1,centroid_len),k = centroids_num,max_iter= 100)
        clusters_list.append(clusters.unsqueeze(0))
        nearest_indices_list.append(nearest_indices.unsqueeze(0))
    clusters_merge = torch.cat(clusters_list,dim = 0)
    nearest_indices_merge = torch.cat(nearest_indices_list,dim = 0)
    return clusters_merge,nearest_indices_merge,scales
class QuantizedWeight(nn.Module):
    EPS = 1e-9

    def __init__(
        self,
        *,
        XTX: torch.Tensor,
        reference_weight: torch.Tensor,
        in_group_size: int,
        out_group_size: int,
        num_codebooks: int,
        nbits_per_codebook: int = 8,
        codebook_value_nbits: int = 16,
        codebook_value_num_groups: int = 1,
        scale_nbits: int = 0,
        straight_through_gradient: Optional[bool] = None,
        rank = 32,
        **init_kwargs,
    ):
        super().__init__()
        device = reference_weight.device
        self.out_features, self.in_features = reference_weight.shape
        assert self.in_features % in_group_size == 0
        assert self.out_features % out_group_size == 0
        self.rows = reference_weight.shape[0]
        self.columns = reference_weight.shape[1]
        self.out_group_size, self.in_group_size = out_group_size, in_group_size
        self.centroid_len = self.in_group_size
        self.num_codebooks = num_codebooks
        self.nbits_per_codebook = nbits_per_codebook
        self.codebook_size = codebook_size = 2**nbits_per_codebook
        self.centroids_num = codebook_size
        self.codebook_value_nbits = codebook_value_nbits
        self.codebook_value_num_groups = codebook_value_num_groups
        self.codebook_value_clusters = None
        self.bolck_size = reference_weight.shape[0]
        self.scales = self.scales_clusters = self.scales_indices = None
        self.rank = rank
        if straight_through_gradient is None and scale_nbits > 0:
            straight_through_gradient = scale_nbits >= 6
        self.straight_through_gradient = straight_through_gradient
        self.scale_nbits = scale_nbits
 #      
        self.L = nn.Parameter(torch.zeros(self.rows,rank).to(device),requires_grad=False)
        self.R = nn.Parameter(torch.zeros(rank,self.columns).to(device),requires_grad=False) 
        clusters_merge,nearest_indices_merge,scales \
            = quantize(reference_weight.float(),codebook_num=num_codebooks,block_size=self.bolck_size,centroid_len=in_group_size)
        self.codebooks = nn.Parameter(clusters_merge,requires_grad=True)
        self.scales = nn.Parameter(scales,requires_grad=True)
        self.codes = nn.Parameter(nearest_indices_merge,requires_grad=False)
        
    def get_codebooks(self) -> torch.Tensor:
        """Get quantization codebooks or reconstruct them from second level quantization (see codebook_values_nbits)"""
        return self.codebooks
        raise NotImplementedError(f"{self.codebook_value_nbits}-bit codebook values are not supported")
    def updateLR(self,weight):
        weight = weight - self.differentiable_dequantize()
        with torch.no_grad():
            output = low_rank_decomposition(weight, reduced_rank=self.rank)
            L, R, reduced_rank = output['L'], output['R'], output['reduced_rank']
            self.L.data=L
            self.R.data=R
    def get_scales(self) -> torch.Tensor:
        return self.scales  # scales are not quantized or the quantization is lossless
    def differentiable_dequantize(self):
        codebook_num = self.num_codebooks
        codes = self.codes.clone().detach()
        for i in range(codebook_num):
            codes[i,:]+=self.centroids_num*i
        # _,count =  torch.unique(codes, return_counts=True)
        # print(count.min(),count.max)
        codebook_offsets = torch.arange(0,(self.rows*self.columns)//self.centroid_len).to(self.codebooks.device)
        reconstruct_weight = F.embedding_bag(codes.flatten(),self.codebooks.flatten(0,1),codebook_offsets,mode="sum")
        cnt = (reconstruct_weight.view(-1,self.bolck_size)*self.scales)
        return cnt.view((self.rows, self.columns))
    
    def forward(self, selection: Union[slice, ellipsis, torch.Tensor] = ...):
        """
        Differentably reconstruct the weight (or parts thereof) from compressed components
        :param selection: By default, reconstruct the entire weight. If selection is specified, this method will instead
            reconstruct a portion of weight for the corresponding output dimensions (used for parallelism).
            The indices / slices must correspond to output channels (if out_group_size==1) or groups (if > 1).
            Formally, the indices must be in range [ 0 , self.out_features // self.out_group_size )

        """
        weight = self.differentiable_dequantize()+torch.mm(self.L, self.R)
        # print(weight.dtype)
        return weight

    def soft_forward(self,weight,scaler_row):
        return
    def estimate_nbits_per_parameter(self) -> float:
        """Calculate the effective number of bits per original matrix parameters"""
        return 0

    def extra_repr(self) -> str:
        return f"{self.out_features=}, {self.in_features=}, bits_per_parameter={self.estimate_nbits_per_parameter()}"

    def update_index(self,weight,scaler_row):
        weight = weight - torch.mm(self.L,self.R)
        shape = weight.shape[0]
        print(self.bolck_size)
        reshspe_weight = weight
        reshspe_weight = reshspe_weight.view(-1,self.bolck_size)
        detach_scales = self.scales.detach()
        normalized_tensor = (reshspe_weight / detach_scales)
        S = scaler_row.unsqueeze(0)\
            .expand(self.rows, -1)\
                .contiguous()\
                    .view(-1,self.bolck_size)
        weight_list = normalized_tensor.split(normalized_tensor.shape[0]//self.num_codebooks,dim = 0)
        S_list = S.split(normalized_tensor.shape[0]//self.num_codebooks,dim = 0)
        nearest_indices_list = []
        index = 0
        for weight,s in zip(weight_list,S_list):
            # search
            nearest_indices=get_nearest_indices(S=s,W = weight.view(-1,self.centroid_len),shape = weight.shape,centroids=self.codebooks[index])
            nearest_indices_list.append(nearest_indices.unsqueeze(0))
            index +=1
        del S
        nearest_indices_merge = torch.cat(nearest_indices_list,dim = 0)
        self.codes.data  =nearest_indices_merge
def get_nearest_indices(
    S: torch.Tensor, #重要性
    W,
    shape, # 权重的原始形状
    centroids,
    devices: Optional[List[torch.device]] = None,
):
    if S is None:
        S = torch.zeros(shape[0]).to(W.device)
        S[0] = 1
        # S[0] = 1
    # if devices is None:
    #     devices = [data.device]
    # W  N*D
    # centroids n_centroids*D
    assignments_list = []
    a1 = W.view(-1,centroids.shape[-1]).unsqueeze(1)
    # S为每一行的重要性权重，将其扩展成矩阵形式，方便计算
    s1 = S.view(-1,centroids.shape[-1]).unsqueeze(1)
    chunks_a = torch.chunk(a1, 2, dim=0)
    chunks_s = torch.chunk(s1, 2, dim=0)
    b1 = centroids.unsqueeze(0)
    for ac,sc in zip(chunks_a,chunks_s):
        dist = ((ac-b1)**2)
        dist = (dist*sc).sum(-1)
        assignments_list.append(dist.argmin(-1))
    
    assignments = torch.cat(assignments_list,dim=0)
    # dist =((a1-b1)**2*s1).sum(-1)
    # assignmentss = dist.argmin(-1)
    # print(assignments - assignmentss)
    # print(assignments.shape)
    return assignments

def soft_dequant(
    S: torch.Tensor, #重要性
    W,
    shape, # 权重的原始形状
    centroids,
    devices: Optional[List[torch.device]] = None,
):
    if S is None:
        S = torch.zeros(shape[0]).to(W.device)
        S[0] = 1
        # S[0] = 1
    # if devices is None:
    #     devices = [data.device]
    # W  N*D
    # centroids n_centroids*D
    assignments_list = []
    a1 = W.view(-1,centroids.shape[-1]).unsqueeze(1)
    # S为每一行的重要性权重，将其扩展成矩阵形式，方便计算
    s1 = S.view(-1,centroids.shape[-1]).unsqueeze(1)
    chunks_a = torch.chunk(a1, 2, dim=0)
    chunks_s = torch.chunk(s1, 2, dim=0)
    b1 = centroids.unsqueeze(0)
    for ac,sc in zip(chunks_a,chunks_s):
        dist = ((ac-b1)**2)
        dist = (dist*sc).sum(-1)
        assignments_list.append(dist.argmin(-1))
    
    assignments = torch.cat(assignments_list,dim=0)
    # dist =((a1-b1)**2*s1).sum(-1)
    # assignmentss = dist.argmin(-1)
    # print(assignments - assignmentss)
    # print(assignments.shape)
    return assignments




@torch.no_grad()
def init_aq_kmeans(
    reference_weight: torch.Tensor,
    *,
    num_codebooks: int,
    out_group_size: int,
    in_group_size: int,
    codebook_size: int,
    verbose: bool = False,
    use_faiss: bool = False,
    max_points_per_centroid: Optional[int] = None,
    max_iter: int = 1000,
    devices: Optional[List[torch.device]] = None,
    **kwargs,
):
    """
    Create initial codes and codebooks using residual K-means clustering of weights
    :params reference_weight, num_codebooks, out_group_size, in_group_size, nbits, verbose: same as in QuantizedWeight
    :params use_faiss  whether to use faiss implementation of kmeans or pure torch
    :params max_point_per_centorid maximum data point per cluster
    :param kwargs: any additional params are forwarded to fit_kmeans
    """
    out_features, in_features = reference_weight.shape
    num_out_groups = out_features // out_group_size
    num_in_groups = in_features // in_group_size
    weight_residue = (
        reference_weight.reshape(num_out_groups, out_group_size, num_in_groups, in_group_size)
        .clone()
        .swapaxes(-3, -2)
        .reshape(num_out_groups * num_in_groups, out_group_size * in_group_size)
    )
    codebooks = []
    codes = []

    if max_points_per_centroid is not None:
        print("Clustering:", max_points_per_centroid * codebook_size, "points from", weight_residue.shape[0])

    for _ in trange(num_codebooks, desc="initializing with kmeans") if verbose else range(num_codebooks):
        if use_faiss:
            codebook_i, codes_i, reconstructed_weight_i = fit_faiss_kmeans(
                weight_residue,
                k=codebook_size,
                max_iter=max_iter,
                gpu=(weight_residue.device.type == "cuda"),
                max_points_per_centroid=max_points_per_centroid,
            )
        else:
            chosen_ids = None
            if max_points_per_centroid is not None:
                chosen_ids = torch.randperm(weight_residue.shape[0], device=weight_residue.device)[
                    : max_points_per_centroid * codebook_size
                ]
            codebook_i, _, _ = fit_kmeans(
                weight_residue if chosen_ids is None else weight_residue[chosen_ids, :],
                k=codebook_size,
                max_iter=max_iter,
                devices=devices,
                **kwargs,
            )
            codes_i, reconstructed_weight_i = find_nearest_cluster(weight_residue, codebook_i, devices=devices)

        codes_i = codes_i.reshape(num_out_groups, num_in_groups, 1)
        codebook_i = codebook_i.reshape(1, codebook_size, out_group_size, in_group_size)
        weight_residue -= reconstructed_weight_i
        codes.append(codes_i)
        codebooks.append(codebook_i)
        del reconstructed_weight_i
    codebooks = torch.cat(codebooks, dim=0)
    codes = torch.cat(codes, dim=-1)
    return codes, codebooks
def low_rank_decomposition(weight, reduced_rank=32):
    """
    :param          weight: The matrix to decompose, of shape (H, W)
    :param    reduced_rank: the final rank
    :return:
    """

    """parameter_ratio = rank * (H + W) / (H * W)"""
    """rank_ratio = """
    matrix_dimension = len(weight.size())
    assert matrix_dimension == 2, "Only Support 2D matrix"
    H, W = weight.size()

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters

    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    rank = torch.count_nonzero(S)
    is_full_rank = rank == min(H, W)

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    # print(f"W: ({H},{W}) | Rank: {rank} | U:{U.shape} | S:{S.shape} | Vh:{Vh.shape}")
    # print(f"Reduced Rank: {reduced_rank} | Num Parameters: {(H + W) * reduced_rank}")
    print(f"L: {L.shape} | R: {R.shape}")

    return {"L": L, "R": R, "U": U, "S": S, "Vh": Vh, 'reduced_rank': reduced_rank}  
