import os, sys
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import torch
from itertools import groupby
from tqdm import tqdm

ONE_PER_SPEAKER = False

try:
    exp_dir = sys.argv[1]
except:
    pass

def extract_id(filename):
    return int(filename.split('_')[0])

f = open("%s/create_d_vectors.log" % exp_dir, "a+")

def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()

printt(sys.argv)

printt(exp_dir)

wavPath = "%s/1_16k_wavs" % exp_dir
outPath = (
    "%s/4_d_vectors" % exp_dir
)
os.makedirs(outPath, exist_ok=True)

wav_fpaths = [f for f in Path(wavPath).glob('*.wav')]

if ONE_PER_SPEAKER:
    speaker_wavs = {speaker: list(map(preprocess_wav, wav_fpaths)) for speaker, wav_fpaths in
                    groupby(tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit="wavs"),
                            key=lambda wav_fpath: extract_id(wav_fpath.name))}
else:
    speaker_wavs = {speaker: list(map(preprocess_wav, wav_fpaths)) for speaker, wav_fpaths in
                    groupby(tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit="wavs"),
                            key=lambda wav_fpath: wav_fpath.stem)}

encoder = VoiceEncoder()
n = max(len(speaker_wavs.keys()) // 5, 1)
for idx, (id_, wavs) in enumerate(speaker_wavs.items()):
    out_path = "%s/%s" % (outPath, str(id_)+".npy")

    if os.path.exists(out_path):
        continue
    
    d_vector = np.asarray([encoder.embed_utterance(wav) for wav in wavs]).mean(axis=0)
    if np.isnan(d_vector).sum() == 0:
        np.save(out_path, d_vector, allow_pickle=False)

    if idx % n == 0:
        printt("now-%s,all-%s" % (idx, len(speaker_wavs.keys())))

printt("all d-vectors done")