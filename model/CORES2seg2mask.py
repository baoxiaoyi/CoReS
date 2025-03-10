from typing import List,Optional
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel
import os
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.relu

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        output = self.norm(tgt)
        return output

def get_expanded_bboxes(mask, new_size):
    batch_size, h, w = mask.size()
    expanded_bboxes = torch.zeros(batch_size, 1, 4)
    
    rate_h,rate_w = new_size/h, new_size/w
    
    for i in range(batch_size):
        bbox = torch.nonzero(mask[i])
        
        if len(bbox) > 0:
            xmin = bbox[:, 0].min().item() * rate_h
            ymin = bbox[:, 1].min().item() * rate_w
            xmax = bbox[:, 0].max().item() * rate_h
            ymax = bbox[:, 1].max().item() * rate_w
            
            # Expand the bbox dimensions
            bbox_h = xmax - xmin
            bbox_w = ymax - ymin
            
            xmin_expanded = max(0, xmin - int(0.2 * bbox_h))
            ymin_expanded = max(0, ymin - int(0.2 * bbox_w))
            xmax_expanded = min(new_size - 1, xmax + int(0.2 * bbox_h))
            ymax_expanded = min(new_size - 1, ymax + int(0.2 * bbox_w))
            
            expanded_bboxes[i, 0, 0] = xmin_expanded
            expanded_bboxes[i, 0, 1] = ymin_expanded
            expanded_bboxes[i, 0, 2] = xmax_expanded
            expanded_bboxes[i, 0, 3] = ymax_expanded
    
    return expanded_bboxes


class CoresMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(CoresMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_cores_modules(self.config)

    def initialize_cores_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        '''self.visual_model2 = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model2.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model2.mask_decoder.train()
            for param in self.visual_model2.mask_decoder.parameters():
                param.requires_grad = True'''

def Weighted_GAP(supp_feat, mask=None):
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    if mask!=None:
        supp_feat = supp_feat * mask
        area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    else:
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:])
    return supp_feat

class CoresModel(CoresMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(CoresModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class CORESForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.seg_sema_token_idx = kwargs.pop("seg_sema_token_idx")

        super().__init__(config)

        self.model = CoresModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.transformer_cross_attention_layers = CrossAttentionLayer(
                    d_model=256,
                    nhead=8,
                    dropout=0.0,
                )
        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,#2,3,1024,1024
        images_clip: torch.FloatTensor,#2,3,224,224
        input_ids: torch.LongTensor,#6,67 完整对话
        labels: torch.LongTensor,#6,67 回答，前面补-100 padding
        attention_masks: torch.LongTensor,#6,67 labels padding部分设置为true
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],#2,H,W
        resize_list: List[tuple],#[[801,1024],[682,1024]]
        inference: bool = False,
        **kwargs,
    ):
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]

        '''import cv2
        import numpy as np
        image = cv2.imread(kwargs["image_paths"][0])
        cv2.imwrite('pascalpart/'+kwargs["image_paths"][0].split('/')[-1], image)
        for p,gt_i in enumerate(masks_list[0]):
                pred_mask_tmp = gt_i.detach().cpu().numpy() == 1
                gt = image.copy()
                gt[pred_mask_tmp] = (image*0.5+pred_mask_tmp[:,:,None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5)[pred_mask_tmp]
                tmp_dir = '0306/'+kwargs["image_paths"][0].split('/')[-1].split('.')[0]+'_gt'+str(p)            
                cv2.imwrite(tmp_dir+'.jpg', gt)'''

        bstext = input_ids.shape[0]
        assert batch_size == len(offset) - 1

        ##########
        seg_sema_token_mask = input_ids[:, 1:] == self.seg_sema_token_idx#6,69
        seg_sema_token_mask = torch.cat(
            [
                seg_sema_token_mask,
                torch.zeros((seg_sema_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )#6,69+False
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_sema_token_mask = torch.cat(
            [torch.zeros((seg_sema_token_mask.shape[0], 255)).bool().cuda(), seg_sema_token_mask],
            dim=1,
        )#6，False*320+70
        ##########

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx#6,69
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )#6,69+False
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )#6，False*320+70

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
            image_embeddings2 = image_embeddings.expand(length, -1, -1, -1).contiguous()
            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            images_clip_list,image_embedding_list = [],[]
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
                image_embedding_i = (
                    image_embeddings[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                image_embedding_list.append(image_embedding_i)
            images_clip_extend = torch.cat(images_clip_list, dim=0)#6,3,224,224
            image_embeddings2 = torch.cat(image_embedding_list,dim=0)

            output = super().forward(
                images=images_clip_extend,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states#len=33

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))#6,321,4096->256

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)#6,400,256]

        ##################
        pred_sema_embeddings = last_hidden_state[seg_sema_token_mask]
        seg_sema_token_counts = seg_sema_token_mask.int().sum(-1)  # [bs, ]

        seg_sema_token_offset = seg_sema_token_counts.cumsum(-1)
        seg_sema_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_sema_token_offset], dim=0
        )

        seg_sema_token_offset = seg_sema_token_offset[offset]

        pred_sema_embeddings_ = []
        for i in range(len(seg_sema_token_offset) - 1):
            start_i, end_i = seg_sema_token_offset[i], seg_sema_token_offset[i + 1]
            pred_sema_embeddings_.append(pred_sema_embeddings[start_i:end_i])
        pred_sema_embeddings = pred_sema_embeddings_

        #################

        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pred_masks_0,low_mask_list = [],[]
        for i in range(len(pred_sema_embeddings)):
            ############2tokenversion
            (
                    sparse_embeddings,
                    dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_sema_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_sema_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
            )
            low_mask_list.append(low_res_masks)
            low_res_masks = low_res_masks.to(pred_sema_embeddings[i].dtype)
            pred_mask_0 = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks_0.append(pred_mask_0[:, 0])

        ###############2seg#######################

        index_list = [False]*bstext
        for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                if start_i==end_i:
                    continue
                for k in range(end_i - start_i):
                    index_list[start_i+k]=True
        square_masks,pred_embedding_ = [],[]
        masked_feat = Weighted_GAP(image_embeddings)
        if any(t.numel()!= 0 for t in low_mask_list):
            target_sizes = [max(mask.shape[1], mask.shape[2]) for mask in low_mask_list]  # 计算最大的尺寸

            for i,(target_size,mask) in enumerate(zip(target_sizes,low_mask_list)):
                if mask.shape[0]==0:
                    continue
                normalized_tensor = (mask>0).to(mask.dtype)
                square_masks.append(normalized_tensor)
                #mask = mask.detach()
                #start,end = seg_token_offset[i],seg_token_offset[i+1]
                '''min_val = torch.min(mask)
                max_val = torch.max(mask)
                normalized_tensor = (mask - min_val) / (max_val- min_val)
                square_masks.append(normalized_tensor)'''
                #images_clip_extend[start:end] = F.interpolate(normalized_tensor, size=(224, 224), mode='bilinear', align_corners=False).expand(-1,3,-1,-1).to(dtype=torch.bfloat16)*images_clip_extend[start:end]
                #image_embeddings2[start:end] = F.interpolate(normalized_tensor, size=(64, 64), mode='bilinear', align_corners=False).expand(-1,256,-1,-1).to(dtype=torch.bfloat16)*image_embeddings2[start:end]

            masked = [F.interpolate(mask, size=(64, 64), mode='bilinear', align_corners=False).expand(-1,256,-1,-1).to(dtype=torch.bfloat16) if mask!=None else mask for mask in square_masks]
            #masked_feat = torch.cat(masked,dim=0)*image_embeddings2[index_list]
            masked_feat = Weighted_GAP(image_embeddings2[index_list],torch.cat(masked,dim=0))
            '''embedding_cat = torch.cat(pred_embeddings,dim=0) + self.transformer_cross_attention_layers(torch.cat(pred_embeddings,dim=0).unsqueeze(0),masked_feat.flatten(2).permute(2,0,1)).permute(1,2,0).squeeze(2)
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embedding_.append(embedding_cat[start_i:end_i])'''
        else:
            #pred_embedding_ = pred_embeddings
            pass
        #pred_embeddings = pred_embedding_

        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
                ##########1tokenversion
                (
                        sparse_embeddings,
                        dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=low_mask_list[i],
                        text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                ############1tokenversion
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                sparse_embeddings = sparse_embeddings+self.transformer_cross_attention_layers(sparse_embeddings.permute(1,0,2),masked_feat[seg_token_offset[i]:seg_token_offset[i+1]].flatten(2).permute(2,0,1)).permute(1,0,2)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=label_list[i].shape,
                )
                pred_masks.append(pred_mask[:, 0])
                if inference:
                    torch.cuda.empty_cache()
        ################2seg_end#####################
        '''for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask_tmp = pred_mask.detach().cpu().numpy()[0]
            pred_mask_tmp = pred_mask_tmp > 0

            save_img = image.copy()
            save_img[pred_mask_tmp] = (
                image * 0.5
                + pred_mask_tmp[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask_tmp]
            tmp_dir = 'pascalpart/'+kwargs["image_paths"][0].split('/')[-1].split('.')[0]+'_dt_'+str(i)
            while os.path.exists(tmp_dir+'.jpg'):
                    tmp_dir += '_0'   
            cv2.imwrite(tmp_dir+'.jpg', save_img)

        for i, pred_mask in enumerate(pred_masks_0):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask_tmp = pred_mask.detach().cpu().numpy()[0]
            pred_mask_tmp = pred_mask_tmp > -1.5

            save_img = image.copy()
            save_img[pred_mask_tmp] = (
                image * 0.5
                + pred_mask_tmp[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask_tmp]
            tmp_dir = 'pascalpart/'+kwargs["image_paths"][0].split('/')[-1].split('.')[0]+'_dt0_'+str(i)
            while os.path.exists(tmp_dir+'.jpg'):
                    tmp_dir += '_0'   
            cv2.imwrite(tmp_dir+'.jpg', save_img)'''


        gt_masks = masks_list
        model_output = output
        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        loss,mask_bce_loss,mask_dice_loss,ce_loss1,ce_loss2,mask_loss = 0,0,0,0,0,0
        mask_bce_loss1 = 0
        mask_dice_loss1 = 0
        num_masks = 0
        
        '''############lagermask as gt for cos_c1##################################
        multimask_output = False
        gt_sema_masks = []
        for i in range(len(masks_list)):
            #points
            if masks_list[i].shape[0]==0:
                gt_sema_masks.append(masks_list[i])
                continue
            coordss = []
            original_height,original_width = masks_list[i].shape[-2],masks_list[i].shape[-1]
            new_size = 64
            padded_mask=[]
            # Determine which side is shorter for padding
            if original_height > original_width:
                    pad_height = 0
                    pad_width = (original_width - original_height)
            else:
                    pad_height = (original_height - original_width)
                    pad_width = 0
            padded_mask = F.pad(masks_list[i], (0, pad_height, 0, pad_width), mode='constant', value=0)
            results = get_expanded_bboxes(padded_mask,new_size).cuda()
            (
                    sparse_embeddings,
                    dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=results,
                    masks=None,
                    text_embeds=None,
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
            )
            low_res_masks = low_res_masks.to(masks_list[i].dtype)
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,#[image.shape[0],image.shape[1]],#
            )
            pm = []
            for p,low in enumerate(pred_mask[:, 0]):
                min_val = torch.min(low.unsqueeze(0))
                max_val = torch.max(low.unsqueeze(0))
                normalized_tensor = (low.unsqueeze(0) - min_val) / (max_val - min_val)
                pm.append(torch.round(normalized_tensor))
            gt_sema_masks.append(torch.cat(pm,dim=0))
        ############lagermask as gt for cos_c1##################################################

        
        for batch_idx in range(len(pred_masks_0)):
            gt_sema_mask = gt_sema_masks[batch_idx]
            pred_mask = pred_masks_0[batch_idx]

            assert (
                gt_sema_mask.shape[0] == pred_mask.shape[0]
            ), "gt_sema_mask.shape: {}, pred_mask.shape: {}".format(
                gt_sema_mask.shape, pred_mask.shape
            )
            mask_bce_loss1 += (
                sigmoid_ce_loss(pred_mask, gt_sema_mask, num_masks=gt_sema_mask.shape[0])
                * gt_sema_mask.shape[0]
            )
            mask_dice_loss1 += (
                dice_loss(pred_mask, gt_sema_mask, num_masks=gt_sema_mask.shape[0])
                * gt_sema_mask.shape[0]
            )
            num_masks += gt_sema_mask.shape[0]

        mask_bce_loss1 = 0.5 * self.bce_loss_weight * mask_bce_loss1 / (num_masks + 1e-8)
        mask_dice_loss1 = 0.5 * self.dice_loss_weight * mask_dice_loss1 / (num_masks + 1e-8)
        ############lagermask as gt for cos_c1##################################################'''

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss2 = 0
        mask_dice_loss2 = 0
        
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss2 += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss2 += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss2 = 2 * self.bce_loss_weight * mask_bce_loss2 / (num_masks + 1e-8)
        mask_dice_loss2 = 2 * self.dice_loss_weight * mask_dice_loss2 / (num_masks + 1e-8)
        mask_bce_loss = mask_bce_loss1 + mask_bce_loss2
        mask_dice_loss = mask_dice_loss1 + mask_dice_loss2
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose t hat there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )
            ################
            seg_sema_token_mask = output_ids[:, 1:] == self.seg_sema_token_idx#6,69
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_sema_token_mask = torch.cat(
                [torch.zeros((seg_sema_token_mask.shape[0], 255)).bool().cuda(), seg_sema_token_mask],
                dim=1,
            )#6，False*320+70
            ################
            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

            ##################
            pred_sema_embeddings = last_hidden_state[seg_sema_token_mask]
            seg_sema_token_counts = seg_sema_token_mask.int().sum(-1)  # [bs, ]

            seg_sema_token_offset = seg_sema_token_counts.cumsum(-1)
            seg_sema_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_sema_token_offset], dim=0
            )

            seg_sema_token_offset = seg_sema_token_offset[offset]

            pred_sema_embeddings_ = []
            for i in range(len(seg_sema_token_offset) - 1):
                start_i, end_i = seg_sema_token_offset[i], seg_sema_token_offset[i + 1]
                pred_sema_embeddings_.append(pred_sema_embeddings[start_i:end_i])
            pred_sema_embeddings = pred_sema_embeddings_

            #################

            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                '''##########1tokenversion
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                ############1tokenversion'''
                ############2tokenversion
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_sema_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(pred_sema_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                
                low_res_masks = low_res_masks.to(pred_sema_embeddings[i].dtype)
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=low_res_masks,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                '''
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                )
                pred_mask = pred_mask.to(pred_sema_embeddings[i].dtype)
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=pred_mask,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                '''############2tokenversion         

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=label_list[i].shape,
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
