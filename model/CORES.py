from typing import List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

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
        batch_size = images.shape[0]
        assert batch_size == len(offset) - 1
        '''import cv2
        import numpy as np
        image = cv2.imread(kwargs["image_paths"][0])
        cv2.imwrite('pacoout/'+kwargs["image_paths"][0].split('/')[-1], image)
        for p,gt_i in enumerate(masks_list[0]):
                heatmap = cv2.applyColorMap(255*gt_i.cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
                gt = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
                tmp_dir = 'pacoout/'+kwargs["image_paths"][0].split('/')[-1].split('.')[0]+'_gt'+str(p)
                while os.path.exists(tmp_dir+'.jpg'):
                    tmp_dir += '_0'                    
                cv2.imwrite(tmp_dir+'.jpg', gt)
        img = F.interpolate(images[0].unsqueeze(0),(256,256))[0].permute(1,2,0)'''
        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

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
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)#6,3,224,224

            output = super().forward(
                images=images_clip,
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
        ##########
        image_embeddings = self.get_visual_embs(images)
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
        pred_masks,pred_masks_0 = [],[]
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
            '''for p,low in enumerate(low_res_masks):
                min_val = torch.min(low.unsqueeze(0))
                max_val = torch.max(low.unsqueeze(0))
                normalized_tensor = (low.unsqueeze(0) - min_val) / (max_val - min_val)
                result22 = (255*normalized_tensor).to(dtype=float).detach().cpu().numpy().astype(np.uint8)
                heatmap = cv2.applyColorMap(result22[0][0], cv2.COLORMAP_JET)
                result2 = cv2.addWeighted(img.to(dtype=float).detach().cpu().numpy().astype(np.uint8),0.7,heatmap,0.3,0)
                tmp_dir = 'sema_gt/'+kwargs["image_paths"][0].split('/')[-1].split('.')[0]+'_dt'+str(2*p)
                while os.path.exists(tmp_dir+'.jpg'):
                    tmp_dir += '_0'                    
                cv2.imwrite(tmp_dir+'.jpg', result22[0][0])
                cv2.imwrite(tmp_dir+'_merge.jpg', result2)'''
            low_res_masks = low_res_masks.to(pred_sema_embeddings[i].dtype)
            pred_mask_0 = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks_0.append(pred_mask_0[:, 0])
            (
                    sparse_embeddings,
                    dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=low_res_masks,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            ############2tokenversion
                
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            '''for p,low in enumerate(low_res_masks):
                min_val = torch.min(low.unsqueeze(0))
                max_val = torch.max(low.unsqueeze(0))
                normalized_tensor = (low.unsqueeze(0) - min_val) / (max_val - min_val)
                result22 = (255*normalized_tensor).to(dtype=float).detach().cpu().numpy().astype(np.uint8)
                heatmap = cv2.applyColorMap(result22[0][0], cv2.COLORMAP_JET)
                result2 = cv2.addWeighted(img.to(dtype=float).detach().cpu().numpy().astype(np.uint8),0.7,heatmap,0.3,0)
                tmp_dir = 'sema_gt/'+kwargs["image_paths"][0].split('/')[-1].split('.')[0]+'_dt'+str(2*p+1)
                while os.path.exists(tmp_dir+'.jpg'):
                    tmp_dir += '_0'                    
                cv2.imwrite(tmp_dir+'.jpg', result22[0][0])
                cv2.imwrite(tmp_dir+'_merge.jpg', result2)'''
            
            low_res_masks = low_res_masks.to(pred_sema_embeddings[i].dtype)
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])
            if inference:
                torch.cuda.empty_cache()

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
            tmp_dir = 'pacoout/'+kwargs["image_paths"][0].split('/')[-1].split('.')[0]+'_dt'+str(i)
            while os.path.exists(tmp_dir+'.jpg'):
                    tmp_dir += '_0'   
            cv2.imwrite(tmp_dir+'.jpg', save_img)

        for i, pred_mask in enumerate(pred_masks_0):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask_tmp = pred_mask.detach().cpu().numpy()[0]
            pred_mask_tmp = pred_mask_tmp > 0

            save_img = image.copy()
            save_img[pred_mask_tmp] = (
                image * 0.5
                + pred_mask_tmp[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask_tmp]
            tmp_dir = 'pacoout/'+kwargs["image_paths"][0].split('/')[-1].split('.')[0]+'_dt2'+str(i)
            while os.path.exists(tmp_dir+'.jpg'):
                    tmp_dir += '_0'   
            cv2.imwrite(tmp_dir+'.jpg', save_img)'''

        model_output = output
        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0

        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
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
