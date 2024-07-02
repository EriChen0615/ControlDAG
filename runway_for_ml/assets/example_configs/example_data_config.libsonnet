local tokenizer_name = "bert-base-uncased";

local MRPC_data_pipeline = {
  name: 'MRPCDataPipeline',
  regenerate: false,
  do_inspect: true,
  transforms: {
    'input:LoadMRPCData': {
      transform_name: 'LoadMRPCDataset',
      setup_kwargs: {},
    },
    'output:BertTokenizeMRPC': {
      input_node: 'input:LoadMRPCData',
      transform_name: 'TokenizeMRPC',
      regenerate: false,
      cache: true,
    },
  }, 
};

{
    MRPC_datapipeline: MRPC_data_pipeline
}