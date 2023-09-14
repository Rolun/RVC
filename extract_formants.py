import parselmouth
from my_utils import load_audio
import numpy as np
import os, sys, traceback, math
import logging
from scipy.interpolate import interp1d

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

        self.formant_bin = 2048 #About half of the reasonable max F4
        self.formant_max = 7500.0#5500.0 #Give some extra room for when we are scaling it
        self.formant_min = 50.0
        self.formant_mel_min = 1127 * np.log(1 + self.formant_min / 700)
        self.formant_mel_max = 1127 * np.log(1 + self.formant_max / 700)

    def get_formant_values(self, formant_obj, track, p_len=None):
        formant_values = []
        for time_step in formant_obj.xs():
            value = formant_obj.get_value_at_time(track,time_step)
            if math.isnan(value):
                value = 0
            formant_values.append(value)

        formant_values = np.asarray(formant_values)
        if len(formant_values[formant_values == 0])>0 and len(formant_values[formant_values != 0]) > 0:
            x_values = np.arange(len(formant_values))
            interpolator = interp1d(x_values[formant_values != 0], formant_values[formant_values != 0], kind='linear', fill_value=0, bounds_error=False)
            formant_values = interpolator(x_values)

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
        f4 = self.get_formant_values(formant, 4, p_len)
        f5 = self.get_formant_values(formant, 5, p_len)

        # import pdb; pdb.set_trace()

        return f1, f2, f3, f4, f5
    
    def coarse_formant(self, fN):
        formant_mel = 1127 * np.log(1 + fN / 700)
        formant_mel[formant_mel > 0] = (formant_mel[formant_mel > 0] - self.formant_mel_min) * (
            self.formant_bin - 2
        ) / (self.formant_mel_max - self.formant_mel_min) + 1

        # use 0 or 1
        formant_mel[formant_mel <= 1] = 1
        formant_mel[formant_mel > self.formant_bin - 1] = self.formant_bin - 1
        formant_coarse = np.rint(formant_mel).astype(int)
        assert formant_coarse.max() <= self.formant_bin-1 and formant_coarse.min() >= 1, (
            formant_coarse.max(),
            formant_coarse.min(),
        )
        return formant_coarse
    
    def go(self, paths):
        if len(paths) == 0:
            printt("no-formant-todo")
        else:
            printt("todo-formant-%s" % len(paths))
            n = max(len(paths) // 5, 1)  # 每个进程最多打印5条
            for idx, (inp_path, formants_path, coarse_formants_path) in enumerate(paths):
                try:
                    if idx % n == 0:
                        printt("formanting,now-%s,all-%s,-%s" % (idx, len(paths), inp_path))
                    if (
                        os.path.exists(formants_path + ".npy") == True
                    ):
                        continue
                    featur_pit = self.compute_formants(inp_path)
                    np.save(
                        formants_path,
                        featur_pit,
                        allow_pickle=False,
                    )
                    coarse_pit = [self.coarse_formant(fN) for fN in featur_pit]
                    np.save(
                        coarse_formants_path,
                        coarse_pit,
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
    opt_root1 = "%s/5b_formants" % (exp_dir)
    opt_root2 = "%s/5a_coarse_formants" % (exp_dir)

    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)
    for name in sorted(list(os.listdir(inp_root))):
        inp_path = "%s/%s" % (inp_root, name)
        if "spec" in inp_path:
            continue
        opt_path1 = "%s/%s" % (opt_root1, name)
        opt_path2 = "%s/%s" % (opt_root2, name)
        paths.append([inp_path, opt_path1, opt_path2])

    ps = []

    p = Process(
        target=featureInput.go,
        args=(
            [paths]
        ),
    )
    ps.append(p)
    p.start()