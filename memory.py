import numpy as np
import traceback, sys

class SimpleMemory():
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self._size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self._size = min(self._size+1, self.max_size)

    # buf <obs, act, rew, next_obs, done>    
    def store_buf(self, buf):
        for r in buf:
            self.store(r[0], r[1], r[2], r[3], r[4])

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self._size, size=batch_size)
        return self.obs1_buf[idxs],self.acts_buf[idxs],self.rews_buf[idxs],self.obs2_buf[idxs],self.done_buf[idxs]

    @property
    def size(self):
        return self._size
    
import plyvel
import protobuf3
from gen.memory import MemoryFramePb

class MemoryFrame(MemoryFramePb):
    def __init__(self, obs=None, act=None, rew=None, next_obs=None, done=None, proto_bytes=None):
        self._proto = MemoryFramePb()
        if proto_bytes is not None:
            self._proto.parse_from_bytes(proto_bytes)
            return
        for t in obs:
            self._proto.obs.append(t)
        for t in act:
            self._proto.act.append(t)
        self._proto.rew = rew
        for t in next_obs:
            self._proto.next_obs.append(t)
        self._proto.done = done

    def encode_to_bytes(self):        
        return self._proto.encode_to_bytes()

    def decode(self):
        return np.array(self._proto.obs, dtype=np.float32), \
               np.array(self._proto.act, dtype=np.float32), \
               np.array(self._proto.rew, dtype=np.float32), \
               np.array(self._proto.next_obs, dtype=np.float32), \
               np.array(self._proto.done, dtype=np.float32) 

def comparator(a, b):
    a, b = int(a), int(b)
    if a < b:
        return -1
    if a > b:
        return 1
    return 0
    
class LevelDbMemory():
    def __init__(self, obs_dim, act_dim, db_path='./leveldb_memory', max_size=int(2e4)):
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._db = plyvel.DB(db_path,
                             create_if_missing=True, \
                             comparator=comparator, \
                             comparator_name=b'IntegerComparator')
        with self._db.raw_iterator() as it:
            it.seek_to_last()
            try:
                self._size = int(it.key()) + 1
            except plyvel._plyvel.IteratorInvalidError:
                self._size = 0

        self._size = 0 if self._size is None else int(self._size)
        self.ptr = self._size
        self._sample_batch_size = None
        self._max_size = max_size

    def store(self, obs, act, rew, next_obs, done):
        frame = MemoryFrame(obs, act, rew, next_obs, done)
        self._db.put(str(int(self.ptr)).encode(), frame.encode_to_bytes())
        self.ptr = (self.ptr+1) % self._max_size
        self._size = max(self._size, self.ptr)

    # buf <obs, act, rew, next_obs, done>    
    def store_buf(self, buf):
        with self._db.write_batch() as wb:
            for r in buf:
                frame = MemoryFrame(r[0], r[1], r[2], r[3], r[4])
                wb.put(str(int(self.ptr)).encode(), frame.encode_to_bytes())
                self.ptr = (self.ptr+1) % self._max_size
                self._size = max(self._size, self.ptr)

    def get(self, pos):
        frame = MemoryFrame(proto_bytes=self._db.get(str(pos).encode()))
        return frame.decode()
    
    def allocate_sample_batch(self, batch_size=32):
        self._sample_batch_size = batch_size
        self._obs_buf = np.zeros([batch_size, self._obs_dim], dtype=np.float32)
        self._acts_buf = np.zeros([batch_size, self._act_dim], dtype=np.float32)
        self._rews_buf = np.zeros(batch_size, dtype=np.float32)
        self._next_obs_buf = np.zeros([batch_size, self._obs_dim], dtype=np.float32)
        self._done_buf = np.zeros(batch_size, dtype=np.float32)
        
    def sample_batch(self, batch_size=32):
        assert self.ptr > 0
            
        if batch_size != self._sample_batch_size:
            self.allocate_sample_batch(batch_size)
        idxs = np.random.randint(0, self.size, size=self._sample_batch_size)
        #print("### mem  idxs {}".format(idxs))
        not_found_count=0
        for i, idx in enumerate(idxs):
            proto_bytes = self._db.get(str(idx).encode())
            
            if proto_bytes is None:
                #raise Exception('Invalid idx:' + str(idx) + ' out of ' + str(self.ptr))
                not_found_count += 1
                continue
            frame = MemoryFrame(proto_bytes=proto_bytes)
            obs, acts, rews, next_obs, done = frame.decode()
            self._obs_buf[i] = obs
            self._acts_buf[i] = acts
            self._rews_buf[i] = rews
            self._next_obs_buf[i] = next_obs
            self._done_buf[i] = done
        # debug
        if not_found_count > 10:  
            print('Items not found {}'.format(not_found_count))    
        return self._obs_buf, self._acts_buf, self._rews_buf, self._next_obs_buf, self._done_buf

    def close(self):
        self._db.close()

    @property
    def size(self):
        return self._size

    @property
    def max_size(self):
        return self._max_size
