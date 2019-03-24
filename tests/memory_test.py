import pytest
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


import numpy as np
import shutil
import os.path
from memory import LevelDbMemory

obs_dim=2
act_dim=2
db_path = './testdb'

@pytest.fixture
def level_db_mem():
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    assert not os.path.exists(db_path)    
    return LevelDbMemory(obs_dim=obs_dim, act_dim=act_dim, db_path=db_path)
    
def test_level_db_mem_basic(level_db_mem):
    mem = level_db_mem
    assert mem.size == 0
    
    mem.store([0.1, 0.2], [0.11, 0.22], 0, [0.111, 0.222], False)
    mem.store([0.2, 0.3], [0.22, 0.33], 1, [0.222, 0.333], False)
    assert mem.size == 2

    last = mem.get(1)
    assert np.allclose(last[0],[0.2, 0.3])
    assert np.allclose(last[1],[0.22, 0.33])
    assert np.allclose(last[2],[1])
    assert np.allclose(last[3],[0.222, 0.333])
    assert np.allclose(last[4],[0])

def test_level_db_mem_sample_batch(level_db_mem):    
    mem = level_db_mem
    assert mem.size == 0
    
    mem.store([0.1, 0.2], [0.11, 0.22], 5, [0.222, 0.333], True)
    assert mem.size == 1

    batch = mem.sample_batch(batch_size=3)
    assert np.allclose(batch[0],[[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]])
    assert np.allclose(batch[1],[[0.11, 0.22], [0.11, 0.22], [0.11, 0.22]])
    assert np.allclose(batch[2],[[5], [5], [5]])
    assert np.allclose(batch[3],[[0.222, 0.333], [0.222, 0.333], [0.222, 0.333]])
    assert np.allclose(batch[4],[[1], [1], [1]])

def test_level_db_mem_max_size(level_db_mem):    
    mem = level_db_mem
    assert mem.size == 0

    for i in range(mem.max_size + 20):
        mem.store([0.1, 0.2], [0.11, 0.22], 5, [0.222, 0.333], True)
        assert mem.size == min(i + 1, mem.max_size-1)

def test_level_db_mem_re_open(level_db_mem):    
    mem = level_db_mem
    assert mem.size == 0

    expected_size = 100
    for i in range(expected_size):
        mem.store([0.1, 0.2], [0.11, 0.22], 5, [0.222, 0.333], True)
        assert mem.size == min(i + 1, mem.max_size-1)
    assert mem.size == expected_size
        
    level_db_mem.close() 
    
    mem = LevelDbMemory(obs_dim=obs_dim, act_dim=act_dim, db_path=db_path)
    assert mem.size == expected_size
    
#level_db_mem()
