#!/usr/bin/python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# Tool to calculate to embed a text file
# The functions can be also imported into another Python code


import re
import os
import tempfile
import sys
import time
import argparse
import numpy as np
from collections import namedtuple
import logging
import math

import torch
import torch.nn as nn

try:
    import onnx
    import onnxruntime as ort
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except Exception as e:
    sys.stderr.write(f"WARNING: could not import some modules: {e}\n")

import embedding_util

# get environment
assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']

sys.path.append(LASER + '/source/lib')
from text_processing import Token, BPEfastApply

SPACE_NORMALIZER = re.compile("\s+")
Batch = namedtuple('Batch', 'srcs tokens lengths')

def buffered_read(fp, buffer_size):
    buffer = []
    for src_str in fp:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


def buffered_arange(max):
    if not hasattr(buffered_arange, 'buf'):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


# TODO Do proper padding from the beginning
def convert_padding_direction(src_tokens, padding_idx, right_to_left=False, left_to_right=False):
    assert right_to_left ^ left_to_right
    pad_mask = src_tokens.eq(padding_idx)
    if not pad_mask.any():
        # no padding, return early
        return src_tokens
    if left_to_right and not pad_mask[:, 0].any():
        # already right padded
        return src_tokens
    if right_to_left and not pad_mask[:, -1].any():
        # already left padded
        return src_tokens
    max_len = src_tokens.size(1)
    range = buffered_arange(max_len).type_as(src_tokens).expand_as(src_tokens)
    num_pads = pad_mask.long().sum(dim=1, keepdim=True)
    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)
    else:
        index = torch.remainder(range + num_pads, max_len)
    return src_tokens.gather(1, index)


class SentenceEncoder:

    def __init__(self, model_path, max_sentences=None, max_tokens=None, cpu=False,
                 fp16=False, verbose=False, sort_kind='quicksort', run_ort=False,
                 ort_model_path="/tmp/laser.onnx", run_amp=False):
        self.run_amp = run_amp
        self.run_ort = run_ort
        self.fp16 = fp16
        self.cpu = cpu
        self.use_cuda = torch.cuda.is_available() and not self.cpu
#        self.use_cudnn = torch.cudnn.is_available() and self.use_cuda
        self.max_sentences = max_sentences
        self.max_tokens = max_tokens
        self.ort_model_path = ort_model_path

        if self.max_tokens is None and self.max_sentences is None:
            self.max_sentences = 1

        state_dict = torch.load(model_path)
        self.encoder = Encoder(**state_dict['params'])
        self.encoder.load_state_dict(state_dict['model'])

#        self.encoder_script = torch.jit.script(self.encoder, (torch.randint(1, 2, (100, 1), dtype=torch.int64), torch.randint(1, 2, (100,), dtype=torch.int64), torch.tensor(100), self.dummy_h0, self.dummy_c0))

#        print(self.encoder_script.code)

        self.dictionary = state_dict['dictionary']
        self.pad_index = self.dictionary['<pad>']
        self.eos_index = self.dictionary['</s>']
        self.unk_index = self.dictionary['<unk>']

        if self.fp16:
            if not self.use_cuda:
                if self.cpu:
                    sys.stderr.write(f"WARNING: you set FP16 and CPU execution. This configuration is not supported, so FP16 is not going to be applied\n")
                else:
                    sys.stderr.write(f"WARNING: you set FP16 and could not allocate GPU. This configuration is not supported, so FP16 is not going to be applied\n")
            else:
                if verbose:
                    print(" - FP16")

                self.encoder.half()
        if (self.run_amp and (not self.use_cuda or self.run_ort or self.cpu)):
            sys.stderr.write(f"WARNING: you set AMP but is not possible to enable it due to one or more of the following reasons: CUDA is not available, CPU execution, ORT execution, unknown")

            self.run_amp = False
        if self.use_cuda:
            if verbose:
                print(' - transfer encoder to GPU')
            self.encoder.cuda()

        if (self.use_cuda and self.run_ort and self.fp16):
            sys.stderr.write(f"WARNING: if you use GPU, ort and FP16 has been observed that the cumulated error is considerable and you might want to change this configuration\n")

        self.ort_model = None

        if self.run_ort:
            if verbose:
                print(" - using onnxruntime")

        self.sort_kind = sort_kind

    def get_h0_c0(self, bsz=100):
        h0 = torch.zeros(self.encoder.num_layers * 2 if self.encoder.bidirectional else self.encoder.num_layers, bsz, self.encoder.hidden_size)
        c0 = torch.zeros(self.encoder.num_layers * 2 if self.encoder.bidirectional else self.encoder.num_layers, bsz, self.encoder.hidden_size)

        return h0, c0

    def _process_batch(self, batch, verbose=False):
        compare_onnxruntime = False
        tokens = batch.tokens
        lengths = batch.lengths
        h0, c0 = self.get_h0_c0(tokens.size()[0])
        cuda_starter, cuda_ender = None, None
        latency = None

        if self.use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()
            h0 = h0.cuda()
            c0 = c0.cuda()
            cuda_starter, cuda_ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            latency = 0.0
        if self.fp16:
            h0 = h0.to(torch.float16)
            c0 = c0.to(torch.float16)

        self.encoder.eval()

        if self.encoder.left_pad:
            # convert left-padding to right-padding
            tokens = convert_padding_direction(
                tokens,
                self.encoder.padding_idx,
                left_to_right=True,
            )

        if self.ort_model is not None:
            ort_inputs = {self.ort_model.get_inputs()[0].name: tokens.detach().cpu().numpy(),
                          self.ort_model.get_inputs()[1].name: lengths.detach().cpu().numpy(),
                          self.ort_model.get_inputs()[2].name: h0.detach().cpu().numpy(),
                          self.ort_model.get_inputs()[3].name: c0.detach().cpu().numpy()}

            if self.use_cuda:
                cuda_starter.record()

            ort_outs = self.ort_model.run(None, ort_inputs)

            if self.use_cuda:
                cuda_ender.record()

                torch.cuda.synchronize()

                curr_time = cuda_starter.elapsed_time(cuda_ender) / 1000
                latency += curr_time

            sentemb = ort_outs[0]

            if compare_onnxruntime:
                if (self.fp16 and not self.use_cuda):
                    sys.stderr.write(f"WARNING: you cannot compare results since FP16 and CPU execution flags are set, which is not supported\n")
                else:
                    sentemb2 = self.encoder(tokens, lengths, h0, c0).detach().cpu().numpy()

                    embedding_util.compare(sentemb, sentemb2, atol=1.0, rtol=1e-4, verbose=True)

        if (self.run_ort and self.ort_model is None):
            if verbose:
                print(" - ort init")

            if not os.path.isfile(self.ort_model_path):
                torch.onnx.export(self.encoder, (tokens, lengths, h0, c0), self.ort_model_path,
                                  input_names=['src_tokens', 'src_lengths', 'h0', 'c0'],
                                  output_names=['sentemb'],
#                                  opset_version=13,
                                  opset_version=12,
                                  export_params=True,
                                  dynamic_axes={'src_tokens': {0: 'batch_size', 1: 'length'},
                                                'src_lengths': {0: 'batch_size'},
                                                'h0': {1: 'batch_size'},
                                                'c0': {1: 'batch_size'},
                                                'sentemb': {0: 'batch_size', 1: 'dim'}}
                              )
                if verbose:
                    print(f" - onnx model stored in '{self.ort_model_path}'")
            else:
                if verbose:
                    print(f" - using onnx model from '{self.ort_model_path}'")

            onnx_model = onnx.load(self.ort_model_path)
            onnx.checker.check_model(onnx_model)

            if verbose:
                print(f" - ort device: {ort.get_device()}")

            ort_providers = [["CPUExecutionProvider"], None]

            if not self.cpu:
                max_bytes_gpu_0 = torch.cuda.get_device_properties(0).total_memory

                if "TensorrtExecutionProvider" in ort.get_available_providers():
#                    ort_providers = [["TensorrtExecutionProvider"], [{'trt_max_workspace_size': str(1073741824 * 2), 'trt_fp16_enable':'False', 'trt_int8_enable': 'False', 'trt_dla_enable': 'False'}]]
                    ort_providers = [["TensorrtExecutionProvider"], None]
                elif "CUDAExecutionProvider" in ort.get_available_providers():
                    ort_providers = [["CUDAExecutionProvider"], None]
                else:
                    print(f" - falling down to CPU in ort since there was not possible to configure another provider")

            if verbose:
                print(f" - using '{ort_providers[0][0]}' as ort provider")

            sess_opt = ort.SessionOptions()
            sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            ort_session = ort.InferenceSession(self.ort_model_path, sess_options=sess_opt, providers=ort_providers[0])

            ort_session.set_providers(*ort_providers)

            if verbose:
                print(f" - ort provider options: {ort_session.get_provider_options()[ort_providers[0][0]]}")

            ort_inputs = {ort_session.get_inputs()[0].name: tokens.detach().cpu().numpy(),
                          ort_session.get_inputs()[1].name: lengths.detach().cpu().numpy(),
                          ort_session.get_inputs()[2].name: h0.detach().cpu().numpy(),
                          ort_session.get_inputs()[3].name: c0.detach().cpu().numpy(),}

            if self.use_cuda:
                cuda_starter.record()

            ort_outs = ort_session.run(None, ort_inputs)

            if self.use_cuda:
                cuda_ender.record()

                torch.cuda.synchronize()

                curr_time = cuda_starter.elapsed_time(cuda_ender) / 1000
                latency += curr_time

            sentemb = ort_outs[0]

            self.ort_model = ort_session

            if compare_onnxruntime:
                if (self.fp16 and not self.use_cuda):
                    sys.stderr.write(f"WARNING: you cannot compare results since FP16 and CPU execution flags are set, which is not supported\n")
                else:
                    sentemb2 = self.encoder(tokens, lengths, h0, c0).detach().cpu().numpy()

                    embedding_util.compare(sentemb, sentemb2, atol=1.0, rtol=1e-4, verbose=True)
        elif (not self.run_ort and self.ort_model is None):
            with torch.cuda.amp.autocast(enabled=self.run_amp):
                if self.use_cuda:
                    cuda_starter.record()

                sentemb = self.encoder(tokens, lengths, h0, c0).detach().cpu().numpy()

                if self.use_cuda:
                    cuda_ender.record()

                    torch.cuda.synchronize()

                    curr_time = cuda_starter.elapsed_time(cuda_ender) / 1000
                    latency += curr_time

        return sentemb, latency

    def _tokenize(self, line):
        tokens = SPACE_NORMALIZER.sub(" ", line).strip().split()
        ntokens = len(tokens)
        ids = torch.LongTensor(ntokens + 1)
        for i, token in enumerate(tokens):
            ids[i] = self.dictionary.get(token, self.unk_index)
        ids[ntokens] = self.eos_index
        return ids

    def _make_batches(self, lines):
        tokens = [self._tokenize(line) for line in lines]
        lengths = np.array([t.numel() for t in tokens])
        indices = np.argsort(-lengths, kind=self.sort_kind)

        def batch(tokens, lengths, indices):
            toks = tokens[0].new_full((len(tokens), tokens[0].shape[0]), self.pad_index)
            for i in range(len(tokens)):
                toks[i, -tokens[i].shape[0]:] = tokens[i]
            return Batch(
                srcs=None,
                tokens=toks,
                lengths=torch.LongTensor(lengths)
            ), indices

        batch_tokens, batch_lengths, batch_indices = [], [], []
        ntokens = nsentences = 0
        for i in indices:
            if nsentences > 0 and ((self.max_tokens is not None and ntokens + lengths[i] > self.max_tokens) or
                                   (self.max_sentences is not None and nsentences == self.max_sentences)):
                yield batch(batch_tokens, batch_lengths, batch_indices)
                ntokens = nsentences = 0
                batch_tokens, batch_lengths, batch_indices = [], [], []
            batch_tokens.append(tokens[i])
            batch_lengths.append(lengths[i])
            batch_indices.append(i)
            ntokens += tokens[i].shape[0]
            nsentences += 1
        if nsentences > 0:
            yield batch(batch_tokens, batch_lengths, batch_indices)

    def encode_sentences(self, sentences, verbose=False):
        indices = []
        results = []
        total_gpu_latency = None

        for batch, batch_indices in self._make_batches(sentences):
            indices.extend(batch_indices)

            r, gpu_latency = self._process_batch(batch, verbose=verbose)

            if gpu_latency is not None:
                if total_gpu_latency is None:
                    total_gpu_latency = gpu_latency
                else:
                    total_gpu_latency += gpu_latency

            results.append(r)

#        if total_gpu_latency is not None:
#            print(f" - GPU latency: {total_gpu_latency}")

        return np.vstack(results)[np.argsort(indices, kind=self.sort_kind)], total_gpu_latency


class Encoder(nn.Module):
    def __init__(
            self, num_embeddings, padding_idx, embed_dim=320, hidden_size=512, num_layers=1, bidirectional=False,
            left_pad=True, padding_value=0.
    ):
        super().__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(num_embeddings, embed_dim, padding_idx=self.padding_idx)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def get_initial_LSTM_state(self, src_tokens, src_lengths):
        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
#        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.detach().cpu())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()

        return bsz, seqlen, x, packed_x, state_size, h0, c0

    def forward(self, src_tokens, src_lengths, h0, c0):
        # embed tokens
        x = self.embed_tokens(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.detach().cpu())

        blayers = self.num_layers

        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        # Set padded outputs to -inf so they are not selected by max-pooling
#        padding_mask = src_tokens.eq(self.padding_idx).t().unsqueeze(-1)
        padding_mask = src_tokens.eq(self.padding_idx).t().unsqueeze(2)

#        if padding_mask.any():
#            x = x.float().masked_fill_(padding_mask, float('-inf')).type_as(x)
        x = x.float().masked_fill_(padding_mask, float('-inf')).type_as(x)

        # Build the sentence embedding by max-pooling over the encoder outputs
        sentemb = x.max(dim=0)[0]

        return sentemb

def EncodeTime(t):
    t = int(time.time() - t)
    if t < 1000:
        print(' in {:d}s'.format(t))
    else:
        print(' in {:d}m{:d}s'.format(t // 60, t % 60))


# Encode sentences (existing file pointers)
def EncodeFilep(encoder, inp_file, out_file, buffer_size=10000, verbose=False, np_savetxt=False):
    n = 0
    t = time.time()
    total_gpu_latency = None

    for sentences in buffered_read(inp_file, buffer_size):
#        encoder.encode_sentences(sentences).tofile(out_file)
        embedding, gpu_latency = encoder.encode_sentences(sentences, verbose=verbose)

        if np_savetxt:
            embedding.resize(embedding.shape[0] * embedding.shape[1])
            np.savetxt(out_file, embedding)
        else:
            embedding.tofile(out_file)

        if gpu_latency is not None:
            if total_gpu_latency is None:
                total_gpu_latency = 0.0

            total_gpu_latency += gpu_latency

        n += len(sentences)
        if verbose and n % 10000 == 0:
            print('\r - Encoder: {:d} sentences'.format(n), end='')
    if verbose:
        print('\r - Encoder: {:d} sentences'.format(n), end='')
        EncodeTime(t)
    if total_gpu_latency is not None:
        print(f" - GPU latency: {total_gpu_latency}")


# Encode sentences (file names)
def EncodeFile(encoder, inp_fname, out_fname,
               buffer_size=10000, verbose=False, over_write=False,
               inp_encoding='utf-8', np_savetxt=False):
    # TODO :handle over write
    if not os.path.isfile(out_fname):
        if verbose:
            print(' - Encoder: {} to {}'.
                  format(os.path.basename(inp_fname) if len(inp_fname) > 0 else 'stdin',
                         os.path.basename(out_fname)))
        fin = open(inp_fname, 'r', encoding=inp_encoding, errors='surrogateescape') if len(inp_fname) > 0 else sys.stdin
        fout = open(out_fname, mode='wb')
        EncodeFilep(encoder, fin, fout, buffer_size=buffer_size, verbose=verbose, np_savetxt=np_savetxt)
        fin.close()
        fout.close()
    elif not over_write and verbose:
        print(' - Encoder: {} exists already'.format(os.path.basename(out_fname)))


# Load existing embeddings
def EmbedLoad(fname, dim=1024, verbose=False):
    x = np.fromfile(fname, dtype=np.float32, count=-1)
    x.resize(x.shape[0] // dim, dim)
    if verbose:
        print(' - Embeddings: {:s}, {:d}x{:d}'.format(fname, x.shape[0], dim))
    return x


# Get memory mapped embeddings
def EmbedMmap(fname, dim=1024, dtype=np.float32, verbose=False):
    nbex = int(os.path.getsize(fname) / dim / np.dtype(dtype).itemsize)
    E = np.memmap(fname, mode='r', dtype=dtype, shape=(nbex, dim))
    if verbose:
        print(' - embeddings on disk: {:s} {:d} x {:d}'.format(fname, nbex, dim))
    return E

def EncodeLoad(args):
    args.buffer_size = max(args.buffer_size, 1)
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    if args.verbose:
        print(f' - Encoder: loading {args.encoder}')

    return SentenceEncoder(args.encoder,
                           max_sentences=args.max_sentences,
                           max_tokens=args.max_tokens,
                           sort_kind='mergesort' if args.stable else 'quicksort',
                           cpu=args.cpu,
                           verbose=args.verbose,
                           run_ort=args.run_ort,
                           fp16=args.fp16,
                           ort_model_path=args.ort_model,
                           run_amp=args.run_amp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LASER: Embed sentences')
    parser.add_argument('--encoder', type=str, required=True,
                        help='encoder to be used')
    parser.add_argument('--token-lang', type=str, default='--',
                        help="Perform tokenization with given language ('--' for no tokenization)")
    parser.add_argument('--bpe-codes', type=str, default=None,
                        help='Apply BPE using specified codes')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Detailed output')

    parser.add_argument('-o', '--output', required=True,
                        help='Output sentence embeddings')
    parser.add_argument('--buffer-size', type=int, default=10000,
                        help='Buffer size (sentences)')
#    parser.add_argument('--max-tokens', type=int, default=12000,
#                        help='Maximum number of tokens to process in a batch')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help='Maximum number of tokens to process in a batch')
    parser.add_argument('--max-sentences', type=int, default=None,
                        help='Maximum number of sentences to process in a batch')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    parser.add_argument('--stable', action='store_true',
                        help='Use stable merge sort instead of quick sort')
    parser.add_argument('--np-savetxt', action='store_true',
                        help='Use np.savetxt instead of np.tofile when storing embedding')
    parser.add_argument('--run-ort', action='store_true',
                        help='Use onnxruntime in order to speed up the inference. The providers order will be TensorRT, CUDA and CPU')
    parser.add_argument('--ort-model', default='/tmp/laser.onnx',
                        help='Where the onnx model will be stored or loaded from')
    parser.add_argument('--fp16', action='store_true',
                        help='Load the model inputs as half (i.e. 16 bits instead of 32 bits)')
    parser.add_argument('--run-amp', action='store_true',
                        help='Run Automatic Mixed Precision')
    args = parser.parse_args()

    if (args.max_tokens is None and args.max_sentences is None):
        args.max_tokens = 12000

    args.buffer_size = max(args.buffer_size, 1)
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    # Load encoder
    encoder = EncodeLoad(args)

    with tempfile.TemporaryDirectory() as tmpdir:
        ifname = ''  # stdin will be used
        if args.token_lang != '--':
            tok_fname = os.path.join(tmpdir, 'tok')
            Token(ifname,
                  tok_fname,
                  lang=args.token_lang,
                  romanize=True if args.token_lang == 'el' else False,
                  lower_case=True, gzip=False,
                  verbose=args.verbose, over_write=False)
            ifname = tok_fname

        if args.bpe_codes:
            bpe_fname = os.path.join(tmpdir, 'bpe')
            BPEfastApply(ifname,
                         bpe_fname,
                         args.bpe_codes,
                         verbose=args.verbose, over_write=False)
            ifname = bpe_fname

        EncodeFile(encoder,
                   ifname,
                   args.output,
                   verbose=args.verbose, over_write=False,
                   buffer_size=args.buffer_size,
                   np_savetxt=args.np_savetxt)
