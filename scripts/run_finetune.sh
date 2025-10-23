#bash scripts/finetune_lora/finetune_lora_1e-5.sh
#bash scripts/finetune_lora/finetune_lora_3e-5.sh
#bash scripts/finetune_lora/finetune_lora_8e-6.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
#bash scripts/finetune_lora_vision/finetune_lora_vision_5e-5.sh
#bash scripts/finetune_lora_vision/finetune_lora_vision_3e-5.sh
#bash scripts/finetune_lora_vision/finetune_lora_vision_5e-5_nomix.sh
#bash scripts/finetune_lora_vision/finetune_lora_vision_3e-5_nomix.sh

bash scripts/finetune_lora/finetune_lora_3e-5.sh
bash scripts/finetune_lora/finetune_lora_1e-5.sh
bash scripts/finetune_lora/finetune_lora_5e-5.sh
