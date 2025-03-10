# CoReS: Orchestrating the Dance of Reasoning and Segmentation

<font size=7><div align='center'><b>CORES</b>: Orchestrating the Dance of Reasoning and Segmentation</div></font>

<font size=7><div align='center' > <a href=https://arxiv.org/abs/2404.05673>**Paper**</a> | <a href="https://chain-of-reasoning-and-segmentation.github.io/">**Project**</a> </div></font>



## News
- [x] [2025.3] Training code is released!
- [x] [2024.7] Paper accepted by ECCV2024!
- [x] [2024.4] [Paper](https://arxiv.org/abs/2404.05673) is released and GitHub repo is created.

**CoReS: Orchestrating the Dance of Reasoning and Segmentation [[Paper](https://arxiv.org/abs/2404.05673)]** <br />

## Abstract
The reasoning segmentation task, which demands a nuanced comprehension of intricate queries to accurately pinpoint object regions, is attracting increasing attention. However, Multi-modal Large Language Models (MLLM) often find it difficult to accurately localize the objects described in complex reasoning contexts. We believe that the act of reasoning segmentation should mirror the cognitive stages of human visual search, where each step is a progressive refinement of thought toward the final object. Thus we introduce the Chains of Reasoning and Segmenting (CoReS) and find this top-down visual hierarchy indeed enhances the visual search process. Specifically, we propose a dual-chain structure that generates multi-modal, chain-like outputs to aid the segmentation process. Furthermore, to steer the MLLM's outputs into this intended hierarchy, we incorporate in-context inputs as guidance. Extensive experiments demonstrate the superior performance of our CoReS, which surpasses the state-of-the-art method by 6.5% on the ReasonSeg dataset. 
For more details, please refer to the [paper](https://arxiv.org/abs/2404.05673).


## Installation
```
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Training
### Training Data Preparation
The training data consists of 4 types of data:

1. Semantic segmentation datasets: [ADE20K](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip), [COCO-Stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip), [Mapillary](https://www.mapillary.com/dataset/vistas), [PACO-LVIS](https://github.com/facebookresearch/paco/tree/main#dataset-setup), [PASCAL-Part](https://github.com/facebookresearch/VLPart/tree/main/datasets#pascal-part), [COCO Images](http://images.cocodataset.org/zips/train2017.zip)

    Note: For COCO-Stuff, we use the annotation file stuffthingmaps_trainval2017.zip. We only use the PACO-LVIS part in PACO. COCO Images should be put into the `dataset/coco/` directory.

3. Referring segmentation datasets: [refCOCO](https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip), [refCOCO+](https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip), [refCOCOg](https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip), [refCLEF](https://web.archive.org/web/20220413011817/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip) ([saiapr_tc-12](https://web.archive.org/web/20220515000000/http://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip)) 

    Note: the original links of refCOCO series data are down, and we update them with new ones. If the download speed is super slow or unstable, we also provide a [OneDrive link](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155154502_link_cuhk_edu_hk/Em5yELVBvfREodKC94nOFLoBLro_LPxsOxNV44PHRWgLcA?e=zQPjsc) to download. **You must also follow the rules that the original datasets require.**

4. Visual Question Answering dataset: [LLaVA-Instruct-150k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json)

5. Reasoning segmentation dataset: [ReasonSeg](https://github.com/dvlab-research/CORES#dataset)

Download them from the above links, and organize them as in LISA.

### Pre-trained weights

#### LLaVA
To train CORES-7B or 13B, you need to follow the [instruction](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) to merge the LLaVA delta weights. Typically, we use the final weights `LLaVA-Lightning-7B-v1-1` and `LLaVA-13B-v1-1` merged from `liuhaotian/LLaVA-Lightning-7B-delta-v1-1` and `liuhaotian/LLaVA-13b-delta-v1-1`, respectively. For Llama2, we can directly use the LLaVA full weights `liuhaotian/llava-llama-2-13b-chat-lightning-preview`.

#### SAM ViT-H weights
Download SAM ViT-H pre-trained weights from the [link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

### Training
```
deepspeed --master_port=24999 train_ds_best.py \
  --version="PATH_TO_LLaVA" \
  --dataset_dir='./dataset' \
  --vision_pretrained="PATH_TO_SAM" \
  --dataset="sem_seg||refer_seg||vqa||reason_seg" \
  --sample_rates="9,3,3,1" \
  --exp_name="cores-7b"
```
When training is finished, to get the full model weight:
```
cd ./runs/cores-7b/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
```

### Merge LoRA Weight
Merge the LoRA weights of `pytorch_model.bin`, save the resulting model into your desired path in the Hugging Face format:
```
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="PATH_TO_LLaVA" \
  --weight="PATH_TO_pytorch_model.bin" \
  --save_path="PATH_TO_SAVED_MODEL"
```

For example:
```
CUDA_VISIBLE_DEVICES="" python3 merge_lora_weights_and_save_hf_model.py \
  --version="Your_Path/model/LLaVA-7B-v0" --lora_r=8\
  --weight="runs/cores-7b/ckpt_model/pytorch_model.bin" \
  --save_path="YOUR_PATH_FOR_CKPT/ckpt/cores7bft"
```

### Validation
```
deepspeed --master_port=10289 train_ds_best.py \
  --version=YOUR_PATH_FOR_CKPT/ckpt/cores7bft \
  --dataset_dir=./dataset \
  --vision_pretrained=YOUR_PATH_FOR_CKPT/ckpt/sam_vit_h_4b8939.pth \
  --dataset="reason_seg" \
  --val_dataset="ReasonSeg|testshort" \
  --sample_rates="1" 
  --exp_name="reasonsegtest" \
  --sample_rates="1" \
  --eval_only
```

## Citation 
If you find this project useful in your research, please consider citing:

```
@inproceedings{bao2024cores,
  title={Cores: Orchestrating the dance of reasoning and segmentation},
  author={Bao, Xiaoyi and Sun, Siyang and Ma, Shuailei and Zheng, Kecheng and Guo, Yuxin and Zhao, Guosheng and Zheng, Yun and Wang, Xingang},
  booktitle={European Conference on Computer Vision},
  pages={187--204},
  year={2024},
  organization={Springer}
}
```

## Acknowledgement
-  This work is built upon the [LISA](https://github.com/dvlab-research/LISA) [LLaVA](https://github.com/haotian-liu/LLaVA) and [SAM](https://github.com/facebookresearch/segment-anything). 
