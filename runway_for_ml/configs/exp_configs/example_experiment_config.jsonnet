local meta_config = import '../meta_config.libsonnet';
local data_config = import '../data_config.libsonnet';

{
    "meta_config": meta_config.meta_config,
    "data_pipeline": data_config.example_data_pipeline,
    "next_data_pipeline": data_config.next_data_pipeline,
    "model_config": {},
    "training": {},
    "validation": {},
    "testing": {},
}