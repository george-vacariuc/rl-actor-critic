import numpy as np
import traceback, sys

class SimpleMemory():
    def __init__(self, obs_dim, act_dim, size, act_dtype=np.float32):
        self._act_dim = act_dim
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=act_dtype)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self._size, self.max_size = 0, 0, size
        self._lifetime_size = 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        assert len(act) == self._act_dim
        insertion_pos = self.ptr
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self._size = min(self._size+1, self.max_size)
        self._lifetime_size = self._lifetime_size + 1
        return insertion_pos

    # buf <obs, act, rew, next_obs, done>
    def store_buf(self, buf):
        for r in buf:
            self.store(r[0], r[1], r[2], r[3], r[4])

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self._size, size=batch_size)
        return self.obs1_buf[idxs],self.acts_buf[idxs],self.rews_buf[idxs],self.obs2_buf[idxs],self.done_buf[idxs]

    @property
    def size(self):
        return self._size

    # Indicates how much experience this memory saw in its entire life.
    @property
    def lifetime_size(self):
        return self._lifetime_size

from sum_tree import SumTree
from threading import Lock
class PriorityMemory(SimpleMemory):
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    PER_b_increment_per_sampling = 0.001
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, obs_dim, act_dim, size, act_dtype):
        SimpleMemory.__init__(self, obs_dim, act_dim, size, act_dtype)
        self.tree = SumTree(size)
        self.tree_lock = Lock()

    def store(self, obs, act, rew, next_obs, done):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        insertion_pos = super().store(obs, act, rew, next_obs, done)
        self.tree_lock.acquire()
        insertion_pos_tree = self.tree.add(max_priority)   # set the max p for new p
        self.tree_lock.release()
        assert insertion_pos == insertion_pos_tree

    def sample_batch(self, batch_size):
        #idxs = np.random.randint(0, self._size, size=batch_size)
        #return self.obs1_buf[idxs],self.acts_buf[idxs],self.rews_buf[idxs],self.obs2_buf[idxs],self.done_buf[idxs]

        mem_idxs, tree_idxs, b_ISWeights =\
            np.empty((batch_size,), dtype=np.int32),\
            np.empty((batch_size,), dtype=np.int32),\
            np.empty((batch_size, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / batch_size       # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        #print('### pp: {}'.format(-self.tree.capacity))
        #print('### pp: {}'.format(self.tree.tree[-self.tree.capacity:]))
        #print('### pp: {}'.format(np.min(self.tree.tree[-self.tree.capacity:])))
        #p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        p_min = self.tree.p_min
        assert p_min > 0
        max_weight = (p_min * batch_size) ** (-self.PER_b)
        assert max_weight > 0

        for i in range(batch_size):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            assert self.tree.data_pointer > 0
            self.tree_lock.acquire()
            index, priority = self.tree.get_leaf(value)
            self.tree_lock.release()
            assert priority > 0, "### index {}".format(index)

            #P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = batch_size * sampling_probabilities
            assert b_ISWeights[i, 0] > 0
            b_ISWeights[i, 0] = np.power(b_ISWeights[i, 0], -self.PER_b)
            b_ISWeights[i, 0] = b_ISWeights[i, 0] / max_weight

            mem_idxs[i]= index - self.max_size + 1
            tree_idxs[i] = index
            #assert b_idx[i] < self.max_size , "{} and {}".format(b_idx[i], self.max_size)
        return self.obs1_buf[mem_idxs],\
            self.acts_buf[mem_idxs],\
            self.rews_buf[mem_idxs],\
            self.obs2_buf[mem_idxs],\
            self.done_buf[mem_idxs],\
            tree_idxs,\
            b_ISWeights

    """
    Update the priorities on the tree
    """
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        self.tree_lock.acquire()
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
        self.tree_lock.release()

import plyvel
import protobuf3
from gen.memory import MemoryFramePb

class MemoryFrame(MemoryFramePb):
    def __init__(self, obs=None, act=None, rew=None, next_obs=None, done=None, proto_bytes=None, memory_frame_pb=None):
        if memory_frame_pb is not None:
            self._proto = memory_frame_pb
            return
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
        proto_bytes=self._db.get(str(pos).encode())
        if proto_bytes is None:
            return None, None, None, None, None
        frame = MemoryFrame(proto_bytes=proto_bytes)
        return frame.decode()

    def allocate_sample_batch(self, batch_size=32):
        self._sample_batch_size = batch_size
        self._obs_buf = np.zeros([batch_size, self._obs_dim], dtype=np.float32)
        self._acts_buf = np.zeros([batch_size, self._act_dim], dtype=np.float32)
        self._rews_buf = np.zeros(batch_size, dtype=np.float32)
        self._next_obs_buf = np.zeros([batch_size, self._obs_dim], dtype=np.float32)
        self._done_buf = np.zeros(batch_size, dtype=np.float32)

    def sample_batch(self, batch_size=32, up_to=None):
        assert self.ptr > 0

        if batch_size != self._sample_batch_size:
            self.allocate_sample_batch(batch_size)
        up_to = self.size if up_to == None else up_to
        idxs = np.random.randint(0, up_to, size=self._sample_batch_size)
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

    def sample_two_consecutive_samples(self, up_to=None):
        assert self.ptr > 0

        # Make of buffer of size 2
        self.allocate_sample_batch(2)
        up_to = self.size if up_to == None else up_to
        # pick a random >= 1
        idx = np.random.randint(1, up_to)
        idxs = [idx - 1, idx]
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


from gen.memory import EpisodeMemoryFramePb

class RedisMemory():
    def __init__(self, redis):
        self.redis = redis
        self.explorer_experience_key = 'explorer_experience_key'

    # buf <obs, act, rew, next_obs, done>
    def store_buf(self, buf):
        explorer_experience = []
        for r in buf:
            frame = MemoryFrame(r[0], r[1], r[2], r[3], r[4])
            explorer_experience.append(frame)
        episodeMemoryFrame = EpisodeMemoryFramePb()
        episodeMemoryFrame.memory_frame.extend(explorer_experience)
        self.redis.rpush(self.explorer_experience_key, episodeMemoryFrame.encode_to_bytes())
        #print('### RedisMemory store_buf Exit.')

    def pull_remote_memory_to_local_memory(self, destination_memory):
        episodeMemoryFrameBytes = self.redis.lpop(self.explorer_experience_key)
        while episodeMemoryFrameBytes is not None:
            episodeMemoryFrame = EpisodeMemoryFramePb()
            try:
                #print('### Before')
                episodeMemoryFrame.parse_from_bytes(episodeMemoryFrameBytes)
                temp_mem = []
                for memory_frame in episodeMemoryFrame.memory_frame:
                    memory_frame = MemoryFrame(memory_frame_pb=memory_frame)
                    current_state, action, reward, new_state, done = memory_frame.decode()
                    #print('### current_state:{} action:{} reward:{} new_state:{} done:{}'.format(current_state, action, reward, new_state, done))

                    temp_mem.append((current_state, action, reward, new_state, done))
                destination_memory.store_buf(temp_mem)
            except:
                print
                continue
            #print('#', end='')
            # Do another one if available.
            episodeMemoryFrameBytes = self.redis.lpop(self.explorer_experience_key)

import zmq
class ZmqMemory():
    def __init__(self):
        PUBSUB_PORT = 5557
        self.PUB = "tcp://*:{}".format(PUBSUB_PORT)
        self.SUB = "tcp://localhost:{}".format(PUBSUB_PORT)
        self._context = zmq.Context()
        self._socket = None


    # buf <obs, act, rew, next_obs, done>
    def store_buf(self, buf):
        if self._socket is None:
            self._socket = self._context.socket(zmq.PUB)
            self._socket.bind(self.PUB)

        explorer_experience = []
        for r in buf:
            frame = MemoryFrame(r[0], r[1], r[2], r[3], r[4])
            explorer_experience.append(frame)
        episodeMemoryFrame = EpisodeMemoryFramePb()
        episodeMemoryFrame.memory_frame.extend(explorer_experience)
        # , zmq.NOBLOCK
        self._socket.send(episodeMemoryFrame.encode_to_bytes())

    def pull_remote_memory_to_local_memory(self, destination_memory):
        if self._socket is None:
            self._socket = self._context.socket(zmq.SUB)
            self._socket.connect(self.SUB)
            self._socket.setsockopt_string(zmq.SUBSCRIBE, '')
            self._socket.setsockopt(zmq.RCVTIMEO, 100)

        try:
            episodeMemoryFrameBytes = self._socket.recv()
        except zmq.Again:
            #traceback.print_exc(file=sys.stdout)
            return
        episodeMemoryFrame = EpisodeMemoryFramePb()
        try:
            episodeMemoryFrame.parse_from_bytes(episodeMemoryFrameBytes)
            temp_mem = []
            for memory_frame in episodeMemoryFrame.memory_frame:
                memory_frame = MemoryFrame(memory_frame_pb=memory_frame)
                current_state, action, reward, new_state, done = memory_frame.decode()
                temp_mem.append((current_state, action, reward, new_state, done))
            destination_memory.store_buf(temp_mem)
        except:
            traceback.print_exc(file=sys.stdout)
