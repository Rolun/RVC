import parselmouth
from my_utils import load_audio
import numpy as np
import os, sys, traceback, math
import logging

now_dir = os.getcwd()
sys.path.append(now_dir)

logging.getLogger("numba").setLevel(logging.WARNING)
from multiprocessing import Process

try:
    exp_dir = sys.argv[1]
    f = open("%s/extract_formants.log" % exp_dir, "a+")
except:
    pass


def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()

class FeatureInput(object):
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

    def get_formant_values(self, formant_obj, track, p_len=None):
        formant_values = []
        for time_step in formant_obj.xs():
            value = formant_obj.get_value_at_time(track,time_step)
            if math.isnan(value):
                value = 0
            formant_values.append(value)

        if p_len:
            pad_size = (p_len - len(formant_values) + 1) // 2
            if pad_size > 0 or p_len - len(formant_values) - pad_size > 0:
                formant_values = np.pad(
                    formant_values, [[pad_size, p_len - len(formant_values) - pad_size]], mode="constant"
                )
        return formant_values

    def compute_formants(self, path, max_number_of_formants=5.5, maximum_formant=5500, pre_emphasis_from=50):
        x = load_audio(path, self.fs)

        p_len = x.shape[0] // self.hop

        time_step = self.hop / self.fs
        formant = (
            parselmouth.Sound(x, self.fs)
            .to_formant_burg(
                time_step=time_step, 
                max_number_of_formants=max_number_of_formants, 
                maximum_formant=maximum_formant, 
                window_length=time_step, 
                pre_emphasis_from=pre_emphasis_from
            )
        )

        f1 = self.get_formant_values(formant, 1, p_len)
        f2 = self.get_formant_values(formant, 2, p_len)
        f3 = self.get_formant_values(formant, 3, p_len)

        # import pdb; pdb.set_trace()

        return f1, f2, f3
    
    def go(self, paths):
        if len(paths) == 0:
            printt("no-formant-todo")
        else:
            printt("todo-formant-%s" % len(paths))
            n = max(len(paths) // 5, 1)  # 每个进程最多打印5条
            for idx, (inp_path, opt_path1) in enumerate(paths):
                try:
                    if idx % n == 0:
                        printt("formanting,now-%s,all-%s,-%s" % (idx, len(paths), inp_path))
                    if (
                        os.path.exists(opt_path1 + ".npy") == True
                    ):
                        continue
                    featur_pit = self.compute_formants(inp_path)
                    np.save(
                        opt_path1,
                        featur_pit,
                        allow_pickle=False,
                    )
                except:
                    printt("formantfail-%s-%s-%s" % (idx, inp_path, traceback.format_exc()))
        print("extract_formant complete")
    
if __name__ == "__main__":
    printt(sys.argv)
    featureInput = FeatureInput()
    paths = []
    inp_root = "%s/1_16k_wavs" % (exp_dir)
    opt_root1 = "%s/5_formants" % (exp_dir)

    os.makedirs(opt_root1, exist_ok=True)
    for name in sorted(list(os.listdir(inp_root))):
        inp_path = "%s/%s" % (inp_root, name)
        if "spec" in inp_path:
            continue
        opt_path1 = "%s/%s" % (opt_root1, name)
        paths.append([inp_path, opt_path1])

    ps = []

    p = Process(
        target=featureInput.go,
        args=(
            [paths]
        ),
    )
    ps.append(p)
    p.start()