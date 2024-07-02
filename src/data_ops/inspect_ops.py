from runway_for_ml.data_module.data_transforms import BaseTransform, HFDatasetTransform, register_transform_functor, keep_ds_columns
from pprint import pprint
import pandas as pd

@register_transform_functor
class InspectSGDDataset(BaseTransform):
    def setup(self, *args, **kwargs):
        pass

    def _call(self, data, *args, **kwargs):
        train_ds = data['train']
        test_ds = data['test']
        train_df = pd.DataFrame(
            data={
                'service': train_ds['service'],
            })
        train_df['split'] = 'train'
        test_df = pd.DataFrame(
            data={
                'service': test_ds['service'],
            })
        test_df['split'] = 'test'
        res_df = pd.concat((train_df, test_df))

        train_service_cnt = train_df.groupby(['service']).count()
        test_service_cnt = test_df.groupby(['service']).count()

        # Show common services in train/test sets
        common_service_df = test_service_cnt.join(train_service_cnt, on='service', lsuffix='_test', rsuffix='_train', how='left').dropna()
        common_service_df['train_test_ratio'] = common_service_df['split_train'] / common_service_df['split_test']
        print(common_service_df)
        return res_df
        
    
    
        
    
