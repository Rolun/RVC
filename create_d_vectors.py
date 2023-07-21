import os, sys
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path


exp_dir = sys.argv[1]

def get_d_vector_resemblyzer(audio_file):
    fpath = Path(audio_file)
    wav = preprocess_wav(fpath)

    encoder = VoiceEncoder()
    embed = encoder.embed_utterance(wav)
    return embed

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

files = sorted(list(os.listdir(wavPath)))
n = max(1, len(files) // 10)

for idx, file in enumerate(files):
    if file.endswith(".wav"):
        wav_path = "%s/%s" % (wavPath, file)
        out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))
        if os.path.exists(out_path):
            continue
        d_vector = get_d_vector_resemblyzer(wav_path)

        if np.isnan(d_vector).sum() == 0:
            np.save(out_path, d_vector, allow_pickle=False)
        else:
            printt("%s-contains nan" % file)
        
        if idx % n == 0:
            printt("now-%s,all-%s,%s,%s" % (len(files), idx, file, d_vector.shape))

printt("all d-vectors done")