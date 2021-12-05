# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.seq2seq
import thumt.models.rnnsearch
import thumt.models.transformer
import thumt.models.multi_heads
import thumt.models.multi_heads_add
import thumt.models.multi_heads_r
import thumt.models.multi_heads_or
def get_model(name):
    name = name.lower()

    if name == "rnnsearch":
        return thumt.models.rnnsearch.RNNsearch
    elif name == "seq2seq":
        return thumt.models.seq2seq.Seq2Seq
    elif name == "transformer":
        return thumt.models.transformer.Transformer
    elif name == 'multi_tasks':
        return thumt.models.multi_heads.Transformer
    elif name == 'add':
        return thumt.models.multi_heads_add.Transformer
    elif name == 'router':
        return thumt.models.multi_heads_r.Transformer
    elif name == 'outr':
        return thumt.models.multi_heads_or.Transformer
    else:
        raise LookupError("Unknown model %s" % name)
