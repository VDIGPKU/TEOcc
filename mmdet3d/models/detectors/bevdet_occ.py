# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet import BEVStereo4D

import torch
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn
import numpy as np
from mmdet3d.models.builder import build_head, build_neck,build_backbone
import random

class Unet3D(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(Unet3D, self).__init__()
        self.init_dres = nn.Conv3d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.hg1 = Hourglass3D(mid_channels)
        self.hg2 = Hourglass3D(mid_channels)

    def forward(self, x):
        dres = self.init_dres(x)
        out1, pre1, post1 = self.hg1(dres, None, None)
        out1 = out1 + dres
        out2, pre2, post2 = self.hg2(out1, pre1, post1)
        out2 = out2 + dres
        return out2

class Hourglass3D(nn.Module):
    def __init__(self, mid_channels):
        super(Hourglass3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, mid_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, 2 * mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, 2 * mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, 2 * mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x, presqu=None, postsqu=None):
        out = self.conv1(x)  # 1 64 10 128 128
        pre = self.conv2(out)  # 1 64 10 128 128

        if postsqu is not None:
            pre = F.leaky_relu(pre + postsqu, inplace=True)
        else:
            pre = F.leaky_relu(pre, inplace=True)
        out = self.conv3(pre)  # 1 64 5 64 64
        out = self.conv4(out)  # 1 64 5 64 64
        out = F.interpolate(out, (pre.shape[-3], pre.shape[-2], pre.shape[-1]), mode='trilinear', align_corners=True)
        out = self.conv5(out)  # 1 64 10 128 128
        if presqu is not None:
            post = F.leaky_relu(out + presqu, inplace=True)
        else:
            post = F.leaky_relu(out + pre, inplace=True)
        out = F.interpolate(post, (x.shape[-3], x.shape[-2], x.shape[-1]), mode='trilinear', align_corners=True)
        out = self.conv6(out)
        return out, pre, post
    
    

@DETECTORS.register_module()
class BEVStereo4DOCC(BEVStereo4D):

    def __init__(self,
                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 with_hop=False,
                 hop_cfg=None,
                 hop_load_all=False,
                 use_short=False,
                 **kwargs):
        super(BEVStereo4DOCC, self).__init__(**kwargs)
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
                        self.img_view_transformer.out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))
        self.use_predicter =use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, num_classes),
            )
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transfromation = False
        
        self.with_hop=with_hop
        if self.with_hop:
            self.hop_cfg=hop_cfg
            self.long_term_backbone=build_backbone(self.hop_cfg.long_term_backbone)
            self.long_term_neck=build_neck(self.hop_cfg.long_term_neck)
            
            self.target_frame=self.hop_cfg.target_frame
        else:
            self.long_term_backbone=False
        self.hop_load_all=hop_load_all
        
        self.use_short=use_short
        if self.use_short:
            self.short_term_decoder=nn.Sequential(
                nn.Conv3d(in_channels=32*2,out_channels=128,kernel_size=3,padding=1),
                nn.ReLU(),
                nn.Conv3d(in_channels=128,out_channels=32,kernel_size=3,padding=1)
            )
            
        
    def loss_single(self,voxel_semantics,mask_camera,preds):
        loss_ = dict()
        voxel_semantics=voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
            loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
            loss_['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics,)
            loss_['loss_occ'] = loss_occ
        return loss_

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    gt_masks_bev=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ ,_= self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        occ_pred = self.final_conv(img_feats[0]).permute(0, 4, 3, 2, 1)
        # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score=occ_pred.softmax(-1)
        occ_res=occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
   
        img_feats, pts_feats, depth,bev_feat_list = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        occ_pred = self.final_conv(img_feats[0])
            
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred.permute(0, 4, 3, 2, 1)) # bncdhw->bnwhdc
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)
        
        if self.hop_load_all:
            num_frames=len(kwargs['hop_voxel_semantics']['semantic'])
            # select_frame=random.randint(0,num_frames-2) #这里为了避免超出范围
            select_frame=random.randint(1,num_frames-1) #这里为了避免超出范围
            gt_semantic=kwargs['hop_voxel_semantics']['semantic'][select_frame]
            gt_mask_camera=kwargs['hop_mask_camera']['mask_camera'][select_frame]
            # bev_feat_list[-1*select_frame-1]=torch.zeros_like(img_feats[0])
            # history_feature=torch.cat(bev_feat_list,dim=1)
            # bev_feat_list[-1*select_frame-1]=torch.zeros_like(img_feats[0])
            
            # history_feature = bev_feat_list[:-1*select_frame-2] + bev_feat_list[-1*select_frame:]
            history_feature = bev_feat_list[:select_frame] + bev_feat_list[select_frame+1:]
            history_feature = torch.cat(history_feature,dim=1)
            # print(select_frame, history_feature.shape)
            if self.long_term_backbone:
                pred_target_feat=self.long_term_backbone(history_feature)
            # if self.long_term_neck:
                pred_target_feat=self.long_term_neck(pred_target_feat)
            
                occ_pred = self.final_conv(pred_target_feat)
                if self.use_predicter:
                    occ_pred = self.predicter(occ_pred.permute(0, 4, 3, 2, 1))
                loss_occ = self.loss_single(gt_semantic, gt_mask_camera, occ_pred)
                losses['random_hop_loss_occ']=loss_occ['loss_occ']
            
            if self.use_short:
                # short_feature=torch.cat([bev_feat_list[-1*select_frame-2],bev_feat_list[-1*select_frame]],dim=1)
                short_feature=torch.cat([bev_feat_list[select_frame-1],bev_feat_list[select_frame+1]],dim=1)
                pred_target_short=self.short_term_decoder(short_feature)
                occ_pred = self.final_conv(pred_target_short)
                if self.use_predicter:
                    occ_pred = self.predicter(occ_pred.permute(0, 4, 3, 2, 1))
                loss_occ = self.loss_single(gt_semantic, gt_mask_camera, occ_pred)

                losses['random_short_loss_occ']=loss_occ['loss_occ']

                
        elif self.with_hop:
            bev_feat_list[self.target_frame]=torch.zeros_like(img_feats[0])
            
            history_feature=torch.cat(bev_feat_list,dim=1)
            
            if self.long_term_backbone:
                pred_target_feat=self.long_term_backbone(history_feature)
            if self.long_term_neck:
                pred_target_feat=self.long_term_neck(pred_target_feat)
            
            occ_pred = self.final_conv(pred_target_feat)
            if self.use_predicter:
                occ_pred = self.predicter(occ_pred.permute(0, 4, 3, 2, 1))
            
            voxel_semantics = kwargs['hop_voxel_semantics']
            mask_camera = kwargs['hop_mask_camera']
            assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17

            loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
            losses['hop_loss_occ']=loss_occ['loss_occ']
        return losses
    
    def extract_feat(self, points, img, img_metas, with_bevencoder=True, **kwargs):
        """Extract features from images and points."""
        img_feats, depth,bev_feat_list = self.extract_img_feat(img, img_metas, with_bevencoder=with_bevencoder, **kwargs)
        pts_feats = None
        return img_feats, pts_feats, depth,bev_feat_list
    
    
    def extract_img_feat(self,
                         img,
                         img_metas,
                         with_bevencoder=True,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            # Todo
            assert False
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = self.prepare_inputs(img, stereo=True)
        """Extract features of images."""
        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None
        for fid in range(self.num_frame - 1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame - self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame)
                if key_frame:
                    bev_feat, depth, feat_curr_iv = \
                        self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv = \
                            self.prepare_bev_feat(*inputs_curr)
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                feat_prev_iv = feat_curr_iv
        if pred_prev:
            # Todo
            assert False
        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) == 4:
                b, c, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1),
                                  h, w]).to(bev_feat_key), bev_feat_key]
            else:
                b, c, z, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1), z,
                                  h, w]).to(bev_feat_key), bev_feat_key]
        if self.align_after_view_transfromation:
            for adj_id in range(self.num_frame - 2):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame - 2 - adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        if with_bevencoder:
            x = self.bev_encoder(bev_feat)
            return [x], depth_key_frame,bev_feat_list
        else:
            return [bev_feat], depth_key_frame,bev_feat_list
