import logging 
from logging import StreamHandler
from logging.handlers import RotatingFileHandler
import os
from collections.abc import Iterable, Mapping
import random
random.seed(1018)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DummyBase: pass
class DataPipelineInspector(DummyBase):
    def __init__(self): pass

    def setup_logger(self, log_dir, maxBytes=20000, backupCount=3):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        rot_file_handler = RotatingFileHandler(
            filename=os.path.join(log_dir, f"test-{self.name}.log"),
            maxBytes=maxBytes,
            backupCount=backupCount,
        )
        console_handler = StreamHandler()
        self.logger.addHandler(rot_file_handler)
        self.logger.addHandler(console_handler)
    
    def setup_inspector(self, config_dict):
        self.do_inspect = True
        self.setup_logger(config_dict['log_dir'])

    def inspect_transform_before(self, transformation_name, transform, outputs):
        self.logger.info(f"{transformation_name}")
        transform_fn = transform.transform_name
        self.logger.info(
f"""
=======================================
Before Transfrom {transform_fn}
*setup_kwargs: {transform.setup_kwargs}
=======================================
"""
        )
        
        pass

    def inspect_transform_after(self, transformation_name, transform, outputs):
        self.logger.info("After Transform")
        pass

    def inspect_loaded_features(self, data):
        # self.logger.info(f"Loaded data: {data}")
        # for split_name, data_split in data.items():
        #     if isinstance(data_split, Iterable):
        #         indices = random.sample(range(len(data_split)), 5)
        #         for i in indices:
        #             self.logger.info(f"Split={split_name}, Index={i}: {data[i]}")
        pass
    

class TestClass(DummyBase):
    def __init__(self, name):
        self.name = name
    
    def run(self):
        if hasattr(self, 'do_inspect'):
            self.log_info('hello! This is '+self.name)




if __name__ == '__main__':
    t1 = TestClass('Eric')   
    extend_instance(t1, DataPipelineInspector)
    t1.setup_logger()
    t1.run()

    
    
