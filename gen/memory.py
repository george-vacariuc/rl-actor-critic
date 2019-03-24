from protobuf3.message import Message
from protobuf3.fields import BoolField, DoubleField


class MemoryFramePb(Message):
    pass

MemoryFramePb.add_field('obs', DoubleField(field_number=1, repeated=True))
MemoryFramePb.add_field('act', DoubleField(field_number=2, repeated=True))
MemoryFramePb.add_field('rew', DoubleField(field_number=3, optional=True))
MemoryFramePb.add_field('next_obs', DoubleField(field_number=4, repeated=True))
MemoryFramePb.add_field('done', BoolField(field_number=5, optional=True))
