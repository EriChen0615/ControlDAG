// This is the project-wide meta configuration file
// It defines project directories, logging options, and other settings shared across different experiments.
// Its value can be override directly in lower-level config files

// Uncomment and set value as appropriate. 

// Default values for training control
local seed=2022;

// data cache configuration
local wandb_cache_dir = "tests/cache/wandb_cache/"; # find out; directory to wandb cache
local default_cache_dir = "tests/cache/"; # find out;

local default_meta = {
  "DATA_FOLDER": "/home/wl356/projects/Knowledge-based-visual-question-answering/data", # find out
  "EXPERIMENT_FOLDER": "./experiments/", # find out
  "TENSORBOARD_FOLDER": "./tensorboards/", 
  "WANDB": {
      "CACHE_DIR":  wandb_cache_dir,
      "entity": "weizhelin",
      "project": "FRAVQA",
      "tags": [],
  },
  "logger_enable": ["tensorboard", "metrics_history"], # "wandb" for wandb logger, "csv" for csv logger
  "platform_type": "pytorch",
  "seed": seed,
  // "cache":{ # find out
  //     "default_dir": default_cache_dir,
  // },
  "default_cache_dir": default_cache_dir,
  "use_versioning": false,
};

{
  default_meta: default_meta
}