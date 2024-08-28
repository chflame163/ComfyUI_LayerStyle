from typing import List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoConfig, AutoModelForCausalLM
from .segment_anything_2.sam2.build_sam import build_sam2
from .unilm.beit3.modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config
from .configuration_evf import EvfConfig
from .segment_anything_2.sam2.utils.misc import load_video_frames
from collections import OrderedDict


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

class EvfSam2Model(PreTrainedModel):
    config_class = EvfConfig
    def __init__(
        self,
        config,
        **kwargs
    ):
        super(EvfSam2Model, self).__init__(config)

        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.encoder_pretrained = kwargs.get("encoder_pretrained", None)
        self.dice_loss_weight = kwargs.get("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.get("bce_loss_weight", None)
        self.train_mask_decoder = kwargs.get("train_mask_decoder", False)
        self.train_prompt_encoder = kwargs.get("train_prompt_encoder", False)
        self.initialize_evf_modules(config)
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    def initialize_evf_modules(self, config):
        # SAM
        if config.sam_scale=="large":
            self.visual_model = build_sam2("sam2_hiera_l.yaml", self.vision_pretrained, device=None)
        elif config.sam_scale=="tiny":
            self.visual_model = build_sam2("sam2_hiera_t.yaml", self.vision_pretrained, device=None)
        else:
            raise NotImplementedError
        
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if self.train_mask_decoder:
            self.visual_model.sam_mask_decoder.train()
            for param in self.visual_model.sam_mask_decoder.parameters():
                param.requires_grad = True
        if self.train_prompt_encoder:
            self.visual_model.sam_prompt_encoder.no_mask_embed.requires_grad_(True)
            
        # beit-3
        if self.config.mm_extractor_scale == "base":
            beit_config = _get_base_config()
        elif self.config.mm_extractor_scale == "large":
            beit_config = _get_large_config()
        else:
            raise AttributeError(f"model config should contain key 'mm_extractor_scale', with value 'base' or 'large'.")

        self.mm_extractor = BEiT3Wrapper(beit_config)
        if self.encoder_pretrained is not None:
            beit_state_dict = torch.load(self.encoder_pretrained)["model"]
            self.mm_extractor.load_state_dict(
                beit_state_dict, 
                strict=False
            )

        for param in self.mm_extractor.parameters():
            param.requires_grad = True
                
        # Projection layer
        in_dim = config.hidden_size
        assert in_dim==beit_config.encoder_embed_dim, \
            f"projection layer dim {in_dim} mismatch with mm_extractor dim {beit_config.encoder_embed_dim}"
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        masks = masks.float()
        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks

    def forward(
        self,
        images: torch.FloatTensor,
        images_evf: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):
        # image_embeddings = self.get_visual_embs(images)     
        backbone_out = self.visual_model.forward_image(images)
        # dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn'])
        _, image_embeddings, _, _ = self.visual_model._prepare_backbone_features(backbone_out)
        image_embeddings = [_.to(images.dtype) for _ in image_embeddings]
        batch_size = images.shape[0]
        if self.visual_model.directly_add_no_mem_embed:
            image_embeddings[-1] = image_embeddings[-1] + self.visual_model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(image_embeddings[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        

        assert batch_size == len(offset) - 1

        images_evf_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            images_evf_i = (
                images_evf[i]
                .unsqueeze(0)
                .expand(end_i - start_i, -1, -1, -1)
                .contiguous()
            )
            images_evf_list.append(images_evf_i)
        images_evf = torch.cat(images_evf_list, dim=0)

        multimask_output = False
        output = self.mm_extractor.beit3(
            visual_tokens=images_evf, 
            textual_tokens=input_ids, 
            text_padding_position=~attention_masks
            )

        feat = output["encoder_out"][:, :1, ...]

        feat = self.text_hidden_fcs[0](feat)
        feat = torch.split(feat, [offset[i+1] - offset[i] for i in range(len(offset)-1)])

        pred_masks = []

        for i in range(len(feat)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.visual_model.sam_prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=feat[i],
            )
            sparse_embeddings = sparse_embeddings.to(feat[i].dtype)
            high_res_features = [
                feat_level[i].unsqueeze(0)
                for feat_level in _features["high_res_feats"]
            ]
            low_res_masks, iou_predictions, _, _ = self.visual_model.sam_mask_decoder(
                image_embeddings=_features["image_embed"][i].unsqueeze(0),
                image_pe=self.visual_model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image = True,
                high_res_features=high_res_features,
            )

            if multimask_output:
                sorted_ids = torch.argsort(iou_predictions, dim=-1, descending=True)
                low_res_masks = torch.take_along_dim(low_res_masks, sorted_ids[..., None, None], dim=1)[:, :1]
          
            pred_mask = self.postprocess_masks(
                low_res_masks,
                orig_hw=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

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

        loss = mask_loss

        return {
            "loss": loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def inference(
            self,
            images,
            images_evf,
            input_ids,
            resize_list,
            original_size_list,
            multimask_output=False,
        ):
        with torch.no_grad():
            backbone_out = self.visual_model.forward_image(images)
            # dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn'])
            _, image_embeddings, _, _ = self.visual_model._prepare_backbone_features(backbone_out)
            image_embeddings = [_.to(images.dtype) for _ in image_embeddings]
            batch_size = images.shape[0]
            if self.visual_model.directly_add_no_mem_embed:
                image_embeddings[-1] = image_embeddings[-1] + self.visual_model.no_mem_embed

            feats = [
                feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                for feat, feat_size in zip(image_embeddings[::-1], self._bb_feat_sizes[::-1])
            ][::-1]
            _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        

        multimask_output = multimask_output

        output = self.mm_extractor.beit3(visual_tokens=images_evf, textual_tokens=input_ids, text_padding_position=torch.zeros_like(input_ids))

        feat = output["encoder_out"][:, :1, ...]
        feat = self.text_hidden_fcs[0](feat)
        (
            sparse_embeddings,
            dense_embeddings,
        ) = self.visual_model.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=feat,
        )
        high_res_features = _features["high_res_feats"]
        sparse_embeddings = sparse_embeddings.to(feat.dtype)
        low_res_masks, iou_predictions, _, _ = self.visual_model.sam_mask_decoder(
            image_embeddings=_features["image_embed"],
            image_pe=self.visual_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image = True,
            high_res_features=high_res_features,
        )
        if multimask_output:
            sorted_ids = torch.argsort(iou_predictions, dim=-1, descending=True)
            low_res_masks = torch.take_along_dim(low_res_masks, sorted_ids[..., None, None], dim=1)[:, :1]

        pred_mask = self.postprocess_masks(
            low_res_masks,
            orig_hw=original_size_list[0],
        )
        
        return pred_mask[:, 0]


AutoConfig.register("evf", EvfConfig)
AutoModelForCausalLM.register(EvfConfig, EvfSam2Model)