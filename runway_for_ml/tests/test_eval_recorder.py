import os
import pickle
import json
import tempfile
import shutil
import copy
from collections import defaultdict
import pytest
import PIL.Image

from runway_for_ml.utils.eval_recorder import EvalRecorder

@pytest.fixture(scope="module")
def tempdir(request):
    # create a temporary directory for testing
    dirpath = "tests/eval_recorder_test"

    def remove_tempdir():
        # remove the temporary directory after testing
        shutil.rmtree(dirpath)

    request.addfinalizer(remove_tempdir)

    return dirpath

def test_init():
    # test the __init__ method of EvalRecorder
    recorder = EvalRecorder(name="test_recorder", base_dir="tests/tmp")
    assert recorder.name == "test_recorder"
    assert recorder.base_dir == "tests/tmp"
    assert recorder.meta_config == {"name": "test_recorder", "base_dir": "tests/tmp"}
    assert recorder._log_index == 0
    assert recorder._sample_logs == defaultdict(list, {'index': []})
    assert recorder._stats_logs == defaultdict(list)

def test_rename():
    # test the rename method of EvalRecorder
    recorder = EvalRecorder(name="test_recorder", base_dir="tests/tmp")
    recorder.rename("new_name")
    assert recorder.name == "new_name"
    assert recorder.meta_config == {"name": "new_name", "base_dir": "tests/tmp"}
    recorder.rename("new_name", "tests/new_dir")
    assert recorder.base_dir == "tests/new_dir"
    assert recorder.meta_config == {"name": "new_name", "base_dir": "tests/new_dir"}

def test_save_to_disk_and_load_from_disk_pkl(tempdir):
    # test the save_to_disk and load_from_disk methods of EvalRecorder
    recorder = EvalRecorder(name="test_recorder", base_dir=tempdir)
    recorder.save_to_disk("eval_recorder", file_format="pkl")
    assert os.path.exists(os.path.join(tempdir, "test_recorder", "eval_recorder.pkl"))
    loaded_recorder = EvalRecorder.load_from_disk("test_recorder", tempdir, "eval_recorder", file_format="pkl")
    assert loaded_recorder.name == "test_recorder"
    assert loaded_recorder.base_dir == tempdir

def test_save_to_disk_and_load_from_disk_json(tempdir):
    # test the save_to_disk and load_from_disk methods of EvalRecorder
    recorder = EvalRecorder(name="test_recorder", base_dir=tempdir)
    file_prefix = "version-1"
    recorder.save_to_disk(file_prefix, file_format="json")
    assert os.path.exists(os.path.join(tempdir, "test_recorder", f"{file_prefix}-sample_log.json"))
    assert os.path.exists(os.path.join(tempdir, "test_recorder", f"{file_prefix}-stats_log.json"))
    assert os.path.exists(os.path.join(tempdir, "test_recorder", f"{file_prefix}-meta_config.json"))
    loaded_recorder = EvalRecorder.load_from_disk("test_recorder", tempdir, "eval_recorder", file_format="pkl")
    assert loaded_recorder.name == "test_recorder"
    assert loaded_recorder.base_dir == tempdir

def test_reset_for_new_pass():
    # test the reset_for_new_pass method of EvalRecorder
    recorder = EvalRecorder(name="test_recorder", base_dir="/tmp")
    recorder.reset_for_new_pass()
    assert recorder._log_index == 0

def test_copy_data_from():
    # test the copy_data_from method of EvalRecorder
    recorder1 = EvalRecorder(name="test_recorder", base_dir="/tmp")
    recorder1._sample_logs["sample"] = [1, 2, 3]
    recorder1._stats_logs["stat"] = [4, 5, 6]
    recorder1.meta_config["config"] = {"foo": "bar"}
    recorder2 = EvalRecorder(name="new_recorder", base_dir="/tmp")
    recorder2.copy_data_from(recorder1)
    assert recorder2._sample_logs == recorder1._sample_logs
    assert recorder2._stats_logs == recorder1._stats_logs
    assert recorder2.meta_config == recorder1.meta_config

def test_log_sample_dict_and_indexing():
    recorder1 = EvalRecorder(name="test_recorder", base_dir="/tmp")
    sample1_dict = {'idx': 1, 'score': 0.2, 'text': "some text"}
    sample2_dict = {'idx': 2, 'score': 0.4, 'text': "some text"}
    sample3_dict = {'idx': 3, 'score': 0.9, 'text': "some text for text 3", 'extra': 'some info'}
    sample4_dict = {'idx': 3, 'score': 0.8, 'text': "some text for text 4"}
    recorder1.log_sample_dict(sample1_dict)
    recorder1.log_sample_dict(sample2_dict)
    recorder1.log_sample_dict(sample3_dict)
    recorder1.log_sample_dict(sample4_dict)
    idx_col = recorder1.get_sample_logs_column('idx')
    index_col = recorder1.get_sample_logs_column('index')
    assert idx_col == [1, 2, 3, 3]
    assert index_col == [0, 1, 2, 3]
    assert recorder1[0] == {'idx': 1, 'index': 0, 'score': 0.2, 'text': 'some text', 'extra': None}
    assert recorder1[2] == {'idx': 3, 'index': 2, 'score': 0.9, 'text': 'some text for text 3', 'extra': 'some info'}
    assert recorder1[3] == {'idx': 3, 'index': 3, 'score': 0.8, 'text': 'some text for text 4', 'extra': None}

def test_log_sample_dict_with_PIL_image():
    recorder1 = EvalRecorder(name="test_recorder", base_dir="test-tmp/")
    img1 = PIL.Image.new(mode='RGB', size=(512, 512))
    img2 = img1.copy()
    img3 = img2.copy()
    sample1_dict = {'score': 0.2, 'text': "some text", 'image': img1}
    sample2_dict = {'score': 0.2, 'text': "some text", 'image': img2}
    sample3_dict = {'score': 0.2, 'text': "some text", 'image': img3}
    recorder1.log_sample_dict(sample1_dict)
    recorder1.log_sample_dict(sample2_dict)
    recorder1.log_sample_dict(sample3_dict)
    image_col = recorder1.get_sample_logs_column('image')
    assert image_col == ['test-tmp/test_recorder/image-0.png', 'test-tmp/test_recorder/image-1.png', 'test-tmp/test_recorder/image-2.png']
    assert os.path.exists(image_col[0])
    assert os.path.exists(image_col[1])
    assert os.path.exists(image_col[2])

def test_log_sample_dict_with_PIL_image_list():
    recorder1 = EvalRecorder(name="test_recorder", base_dir="test-tmp/")
    img1 = PIL.Image.new(mode='RGB', size=(512, 512))
    img2 = img1.copy()
    img3 = img2.copy()
    sample1_dict = {'score': 0.2, 'text': "some text", 'image': [img1, img2, img3]}
    sample2_dict = {'score': 0.2, 'text': "some text", 'image': [img1, img2, img3]}
    recorder1.log_sample_dict(sample1_dict)
    recorder1.log_sample_dict(sample2_dict)

    image_col = recorder1.get_sample_logs_column('image')
    assert image_col == [['test-tmp/test_recorder/image-0-0.png', 'test-tmp/test_recorder/image-0-1.png', 'test-tmp/test_recorder/image-0-2.png'],
                         ['test-tmp/test_recorder/image-1-0.png', 'test-tmp/test_recorder/image-1-1.png', 'test-tmp/test_recorder/image-1-2.png']]
    assert os.path.exists(image_col[0][0])
    assert os.path.exists(image_col[0][1])
    assert os.path.exists(image_col[0][2])
    assert os.path.exists(image_col[1][0])
    assert os.path.exists(image_col[1][1])
    assert os.path.exists(image_col[1][2])
    

def test_log_sample_dict_batch():
    recorder1 = EvalRecorder(name="test_recorder", base_dir="/tmp")
    batch_dict1 = {'idx': [1, 2, 3, 3], 'score': [0.1, 0.2, 0.3, 0.4], 'text': ['t1', 't2', 't3', 't4']}
    sample1_dict = {'idx': 4, 'score': 0.8, 'text': "some text for text 4"}
    batch_dict2 = {'idx': [1, 2, 3, 3], 'text': ['t1', 't2', 't3', 't4']}
    recorder1.log_sample_dict_batch(batch_dict1)
    recorder1.log_sample_dict(sample1_dict)
    recorder1.log_sample_dict_batch(batch_dict2)

    assert recorder1.get_sample_logs_column('idx') == [1, 2, 3, 3, 4 , 1, 2, 3, 3]
    assert recorder1.get_sample_logs_column('score') == [0.1, 0.2, 0.3, 0.4, 0.8, None, None, None, None]
    assert recorder1._log_index == 9
    assert len(recorder1) == 9


def test_reset_for_new_pass_log_sample_dict():
    recorder1 = EvalRecorder(name="test_recorder", base_dir="/tmp")
    sample1_dict = {'idx': 1, 'score': 0.2, 'text': "some text"}
    sample2_dict = {'idx': 2, 'score': 0.4, 'text': "some text"}
    sample3_dict = {'idx': 3, 'score': 0.9, 'text': "some text for text 3", 'extra': 'some info'}
    sample4_dict = {'idx': 3, 'score': 0.8, 'text': "some text for text 4"}
    recorder1.log_sample_dict(sample1_dict)
    recorder1.log_sample_dict(sample2_dict)
    recorder1.log_sample_dict(sample3_dict)
    recorder1.log_sample_dict(sample4_dict)

    recorder1.reset_for_new_pass()
    recorder1.log_sample_dict({'new_score': -0.02})
    recorder1.log_sample_dict({'new_score': -0.03})
    recorder1.log_sample_dict({'new_score': -0.06})
    recorder1.log_sample_dict({'new_score': -0.10})
    assert recorder1.get_sample_logs_column('score') == [0.2, 0.4, 0.9, 0.8]
    assert recorder1.get_sample_logs_column('new_score') == [-0.02, -0.03, -0.06, -0.10]

def test_set_sample_logs_data():
    col_length = 10
    scores = [i*0.1 for i in range(col_length)]
    hypos = [f'hypo text {i}' for i in range(col_length)]
    refs = [f'ref text {i}' for i in range(col_length)]
    data_to_set = {
        'data_id': [i*10 for i in range(col_length)],
        'ref': refs,
        'hypo': hypos,
        'score': scores
    }

    recorder1 = EvalRecorder(name="test_recorder", base_dir="/tmp")
    recorder1.set_sample_logs_data(data_to_set)
    assert recorder1._log_index == col_length
    assert recorder1.get_sample_logs_column('score') == scores
    assert len(recorder1) == col_length

    new_sample_dict = {'data_id': 99, 'ref': 'new ref', 'hypo': 'new hypo', 'score': 1.0}
    recorder1.log_sample_dict(new_sample_dict)
    assert recorder1[col_length] == {'index': 10, 'data_id': 99, 'ref': 'new ref', 'hypo': 'new hypo', 'score': 1.0}

def test_merge_eval_recorders():
    recorder1 = EvalRecorder(name="test_recorder1", base_dir="/tmp")
    sample1_dict = {'idx': 1, 'a-score': 0.2, 'text': "some text"}
    sample2_dict = {'idx': 2, 'a-score': 0.4, 'text': "some text"}
    recorder1.log_sample_dict(sample1_dict)
    recorder1.log_sample_dict(sample2_dict)
    recorder1.log_stats_dict({'a-score-avg': 0.3})
    assert len(recorder1) == 2
    
    recorder2 = EvalRecorder(name="test_recorder2", base_dir="/tmp")
    sample3_dict = {'idx': 1, 'b-score': 0.6, 'text': "some text"}
    sample4_dict = {'idx': 2, 'b-score': 0.8, 'text': "some text"}
    recorder2.log_sample_dict(sample3_dict)
    recorder2.log_sample_dict(sample4_dict)
    recorder2.log_stats_dict({'b-score-avg': 0.7})
    assert len(recorder2) == 2
    
    recorder1.merge([recorder2])
    recorder1.rename('merged_recorder')
    assert recorder1.get_stats_logs() == {'a-score-avg': 0.3, 'b-score-avg': 0.7}
    assert recorder1.get_sample_logs_column('b-score') == [0.6, 0.8]
    

    

