from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags


flags.DEFINE_string("src_dir", "./data/src_aligned", "src_aligned_face_path")
flags.DEFINE_string("dst_dir", "./data/dst_aligned", "dst_aligned_face_path")
flags.DEFINE_integer("batch_size", 1, "batch size")