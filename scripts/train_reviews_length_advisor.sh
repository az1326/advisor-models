#!/bin/bash

# Training script for reviews length domain

# Set environment variables
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export PYTHONPATH="/advisor-models/SkyRL/skyrl-train:$PYTHONPATH"
export DATA_DIR="/advisor-models/data/reviews"
export NUM_GPUS=8
export LOGGER="wandb"  # change to "console" to print to stdout

# Run training
/advisor-models/SkyRL/skyrl-train/.venv/bin/python -m advisor_models.reviews.main_reviews_length \
  data.train_data="['$DATA_DIR/train_length.parquet']" \
  data.val_data="['$DATA_DIR/validation_length.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-7B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=5 \
  trainer.eval_batch_size=100 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.ckpt_interval=999999 \
  trainer.max_prompt_length=8192 \
  generator.sampling_params.max_generate_length=16384 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=reviews \
  generator.n_samples_per_prompt=8 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="advisor_models" \
  trainer.run_name="reviews_length" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/reviews_length"
