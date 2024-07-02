// This is the project-wide meta configuration file
// It defines project directories, logging options, and other settings shared across different experiments.
// Its value can be override directly in lower-level config files

// Uncomment and set value as appropriate. 

// Default values for training control
local seed=2022;

// data cache configuration
local wandb_cache_dir = "./wandb_cache/"; # find out; directory to wandb cache
local default_cache_dir = "./local_cache/"; # find out;

local meta_config = {
  "DATA_FOLDER": "", # find out
  "EXPERIMENT_FOLDER": "./experiments/", # find out
  "TENSORBOARD_FOLDER": "./tensorboards/", 
  "WANDB": {
      "CACHE_DIR":  wandb_cache_dir,
      "USER_NAME": "YOUR WANDB USER NAME HERE",
      "PROJECT_NAME": "YOUR WANDB PROJECT NAME HERE",
  },
  "platform_type": "pytorch",
  "ignore_pretrained_weights": [],
  "seed": seed,
  // "cache":{ # find out
  //     "default_dir": default_cache_dir,
  // },
  "default_cache_dir": default_cache_dir,
  "cuda": 0,
  "gpu_device":0,
};

{
  meta_config: meta_config
}