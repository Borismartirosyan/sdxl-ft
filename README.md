# Stable Diffusion XL fine tuning with advanced fine tunning of Dreambooth and LoRA by Diffusers

This repository is created by to fine-tune a SDXL model with diffusers library. They suggest the train_dreambooth_lora_sdxl_advanced.py framework which is the main tool for us. First, we will start by nessecary installations. 

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install numpy=1.24.4 
conda install xformers -c xformers
pip install bitsandbytes transformers accelerate wandb dadaptation prodigyopt peft triton
pip install datasets
pip install pillow==9.4.0
```
In Pytorch version website ensure that you install corresponding We need a specific version of numpy and pillow to avoid errors with pytorch and xformers. Then we need to install diffusers in editable mode (I pushed it to this repo, you can use my diffusers if you want, else, delete it and use newer versions).

```
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
```
To train the model we need to create the dataset for diffusers, with their desired format. Initially I assume that you have the pictures. 
For that we firstly go to https://aistudio.google.com/app/apikey and get gemini API key. Then we make it as Env variable like this:

```
export GOOGLE_API_KEY=<YOUR_API_KEY>
```

Then we need to caption them. For that we go to scripts. and run:

```
python3 llm_client.py
```

Then we go to /examples/advanced_diffusion_training/ and here is my example code of run how I fine tunned a model. 

```
accelerate launch train_dreambooth_lora_sdxl_advanced.py \
  --pretrained_model_name_or_path="SG161222/RealVisXL_V4.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --dataset_name="/home/jupyter/sdxl-ft/data/instance_data_dir" \
  --instance_prompt="a zwx sculpture in the style of zwx" \
  --validation_prompt="a zwx sculpture boy holding playstation in his hands in the style of zwx" \
  --output_dir="/home/jupyter/sdxl/" \
  --caption_column="prompt" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=3 \
  --repeats=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1.0 \
  --text_encoder_lr=1.0 \
  --adam_beta2=0.99 \
  --optimizer="prodigy"\
  --train_text_encoder_ti\
  --train_text_encoder_ti_frac=0.5\
  --snr_gamma=5.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank=8 \
  --max_train_steps=1000 \
  --checkpointing_steps=500 \
  --seed="0" \
  --token_abstraction "zwx" \
```
You can see sample of results in the output_examples folder.

The inference you can see in infer.py file. An example of usage: 

```
python3 infer.py --pretrained_lora_local_path "checkpoints/pytorch_lora_weights.safetensors" --output_image_path "out.jpg"
```

In replicate.txt file you can see some of the pictures examples how the prompt and negative prompt are created, which seed and lora alpha to use to have the mentioned picture.

