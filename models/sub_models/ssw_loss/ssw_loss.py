import torch
import torch.nn as nn
import torch.nn.functional as F
import kmeans1d

class StereoWhiteningLoss(object):
    def __init__(self):

        self.eps = 1e-5
        self.relax_denom = 1.5
        self.clusters = 3
        self.num_off_diagonal = 0
        self.margin = 0


    def __call__(self, l_w_arr, cov_list, weight=0.6):
        wt_loss  = 0
        for idx in range(len(l_w_arr)):
            feats_l = l_w_arr[idx].transpose(1,2)
            device = feats_l.device

            B, c, num = feats_l.size()
            dim = c

            eye = torch.eye(c, c).cuda()
            reversal_eye = torch.ones(c, c).triu(diagonal=1).cuda()

            cov_matrix = cov_list[idx]

            var_flatten = cov_matrix.flatten()
            clusters, centroids = kmeans1d.cluster(var_flatten, self.clusters)
            num_sensitive = clusters.count(self.clusters-1)
            values, indices = torch.topk(var_flatten, k=int(num_sensitive))

            mask_matrix = torch.zeros(B, dim, dim).cuda()
            mask_matrix = mask_matrix.view(B, -1).contiguous()
            for midx in range(B):
                mask_matrix[midx][indices] = 1
            mask_matrix = mask_matrix.view(B, dim, dim).contiguous()
            mask_matrix = mask_matrix * reversal_eye
            num_sensitive_sum = torch.sum(mask_matrix)
            loss = self.instance_whitening_loss(cov_matrix, mask_matrix, num_remove_cov=num_sensitive_sum)
            wt_loss += loss
        
        wt_loss = wt_loss / len(l_w_arr)


        return wt_loss
    
    def cal_cov(self, raw_w_arr):
        cov_list = []
        l_arr_mask = raw_w_arr[0]
        r_arr_mask = raw_w_arr[1]
        for idx in range(len(l_arr_mask)):
            mask_feats_l = l_arr_mask[idx].transpose(1,2)
            mask_feats_r = r_arr_mask[idx].transpose(1,2)
            b, c, num_points = mask_feats_l.size()   
            eye = torch.eye(c, c).cuda()
            reversal_eye = torch.ones(c, c).triu(diagonal=1).cuda()
            f_map = torch.cat([mask_feats_l.unsqueeze(0), mask_feats_r.unsqueeze(0)], dim=0)
            V, B, C, NUM = f_map.shape   
            f_map = f_map.contiguous().view(V*B, C, -1)
            f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(NUM - 1) + (self.eps * eye) 
            off_diag_elements = f_cor
            off_diag_elements = off_diag_elements.view(V, B, C, -1)
            f_cor = f_cor.view(V, B, C, -1)
            assert V == 2
            variance_of_covariance = torch.var(off_diag_elements, dim=0)
            variance_of_covariance = torch.sum(variance_of_covariance, dim=0)/B
            cov_list.append(variance_of_covariance)
        
        return cov_list
    
    def instance_whitening_loss(self, f_cor, mask_matrix, num_remove_cov):
        f_cor_masked = f_cor * mask_matrix
        off_diag_sum = torch.sum(torch.abs(f_cor_masked), dim=(1,2), keepdim=True) # B X 1 X 1
        loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0) # B X 1 X 1
        loss = torch.sum(loss)
        
        return loss
