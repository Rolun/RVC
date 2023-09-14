import numpy as np, parselmouth, torch, pdb
from time import time as ttime
import torch.nn.functional as F
import torchcrepe # Fork feature. Use the crepe f0 algorithm. New dependency (pip install torchcrepe)
from torch import Tensor
import scipy.signal as signal
import pyworld, os, traceback, faiss, librosa, torchcrepe
from scipy import signal
from functools import lru_cache
import math
from scipy.interpolate import interp1d

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

input_audio_path2wav = {}

@lru_cache
def cache_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    audio = input_audio_path2wav[input_audio_path]
    f0, t = pyworld.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0


def change_rms(data1, sr1, data2, sr2, rate):  # 1是输入音频，2是输出音频,rate是2的占比
    # print(data1.max(),data2.max())
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)

    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()

    return data2


class VC(object):
    def __init__(self, tgt_sr, config):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.sr = 16000  # hubert输入采样率
        self.window = 160  # 每帧点数
        self.t_pad = self.sr * self.x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # 查询切点前后查询时间
        self.t_center = self.sr * self.x_center  # 查询切点位置
        self.t_max = self.sr * self.x_max  # 免查询时长阈值
        self.device = config.device

    # Fork Feature: Get the best torch device to use for f0 algorithms that require a torch device. Will return the type (torch.device)
    def get_optimal_torch_device(self, index: int = 0) -> torch.device:
        # Get cuda device
        if torch.cuda.is_available():
            return torch.device(f"cuda:{index % torch.cuda.device_count()}") # Very fast
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        # Insert an else here to grab "xla" devices if available. TO DO later. Requires the torch_xla.core.xla_model library
        # Else wise return the "cpu" as a torch device, 
        return torch.device("cpu")

    # Fork Feature: Compute f0 with the crepe method
    def get_f0_crepe_computation(
            self, 
            x, 
            f0_min,
            f0_max,
            p_len,
            hop_length=160, # 512 before. Hop length changes the speed that the voice jumps to a different dramatic pitch. Lower hop lengths means more pitch accuracy but longer inference time.
            model="full", # Either use crepe-tiny "tiny" or crepe "full". Default is full
    ):
        x = x.astype(np.float32) # fixes the F.conv2D exception. We needed to convert double to float.
        x /= np.quantile(np.abs(x), 0.999)
        torch_device = self.get_optimal_torch_device()
        audio = torch.from_numpy(x).to(torch_device, copy=True)
        audio = torch.unsqueeze(audio, dim=0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True).detach()
        audio = audio.detach()
        print("Initiating prediction with a crepe_hop_length of: " + str(hop_length))
        pitch: Tensor = torchcrepe.predict(
            audio,
            self.sr,
            hop_length,
            f0_min,
            f0_max,
            model,
            batch_size=hop_length * 2,
            device=torch_device,
            pad=True
        )
        p_len = p_len or x.shape[0] // hop_length
        # Resize the pitch for final f0
        source = np.array(pitch.squeeze(0).cpu().float().numpy())
        source[source < 0.1] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source
        )
        f0 = np.nan_to_num(target)
        return f0 # Resized f0
    
    def get_f0_official_crepe_computation(
            self,
            x,
            f0_min,
            f0_max,
            model="full",
    ):
        # Pick a batch size that doesn't cause memory errors on your gpu
        batch_size = 512
        # Compute pitch using first gpu
        audio = torch.tensor(np.copy(x))[None].float()
        f0, pd = torchcrepe.predict(
            audio,
            self.sr,
            self.window,
            f0_min,
            f0_max,
            model,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0].cpu().numpy()
        return f0

    # Fork Feature: Compute pYIN f0 method
    def get_f0_pyin_computation(self, x, f0_min, f0_max):
        y, sr = librosa.load('saudio/Sidney.wav', self.sr, mono=True)
        f0, _, _ = librosa.pyin(y, sr=self.sr, fmin=f0_min, fmax=f0_max)
        f0 = f0[1:] # Get rid of extra first frame
        return f0

    # Fork Feature: Acquire median hybrid f0 estimation calculation
    def get_f0_hybrid_computation(
        self, 
        methods_str, 
        input_audio_path,
        x,
        f0_min,
        f0_max,
        p_len,
        filter_radius,
        crepe_hop_length,
        time_step,
    ):
        # Get various f0 methods from input to use in the computation stack
        s = methods_str
        s = s.split('hybrid')[1]
        s = s.replace('[', '').replace(']', '')
        methods = s.split('+')
        f0_computation_stack = []

        print("Calculating f0 pitch estimations for methods: %s" % str(methods))
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        # Get f0 calculations for all methods specified
        for method in methods:
            f0 = None
            if method == "pm":
                f0 = (
                    parselmouth.Sound(x, self.sr)
                    .to_pitch_ac(
                        time_step=time_step / 1000,
                        voicing_threshold=0.6,
                        pitch_floor=f0_min,
                        pitch_ceiling=f0_max,
                    )
                    .selected_array["frequency"]
                )
                pad_size = (p_len - len(f0) + 1) // 2
                if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                    f0 = np.pad(
                        f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                    )
            elif method == "crepe":
                f0 = self.get_f0_official_crepe_computation(x, f0_min, f0_max)
                f0 = f0[1:] # Get rid of extra first frame
            elif method == "crepe-tiny":
                f0 = self.get_f0_official_crepe_computation(x, f0_min, f0_max, "tiny")
                f0 = f0[1:] # Get rid of extra first frame
            elif method == "mangio-crepe":
                f0 = self.get_f0_crepe_computation(x, f0_min, f0_max, p_len, crepe_hop_length)
            elif method == "mangio-crepe-tiny":
                f0 = self.get_f0_crepe_computation(x, f0_min, f0_max, p_len, crepe_hop_length, "tiny")
            elif method == "harvest":
                f0 = cache_harvest_f0(input_audio_path, self.sr, f0_max, f0_min, 10)
                if filter_radius > 2:
                    f0 = signal.medfilt(f0, 3)
                f0 = f0[1:] # Get rid of first frame.
            elif method == "dio": # Potentially buggy?
                f0, t = pyworld.dio(
                    x.astype(np.double),
                    fs=self.sr,
                    f0_ceil=f0_max,
                    f0_floor=f0_min,
                    frame_period=10
                )
                f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sr)
                f0 = signal.medfilt(f0, 3)
                f0 = f0[1:]
            #elif method == "pyin": Not Working just yet
            #    f0 = self.get_f0_pyin_computation(x, f0_min, f0_max)
            # Push method to the stack
            f0_computation_stack.append(f0)
        
        for fc in f0_computation_stack:
            print(len(fc))

        print("Calculating hybrid median f0 from the stack of: %s" % str(methods))
        f0_median_hybrid = None
        if len(f0_computation_stack) == 1:
            f0_median_hybrid = f0_computation_stack[0]
        else:
            f0_median_hybrid = np.nanmedian(f0_computation_stack, axis=0)
        return f0_median_hybrid

    def coarse_formant(self, fN):
        formant_bin = 512
        formant_max = 5500.0
        formant_min = 50.0
        formant_mel_min = 1127 * np.log(1 + formant_min / 700)
        formant_mel_max = 1127 * np.log(1 + formant_max / 700)

        fN[fN<0]=0

        formant_mel = 1127 * np.log(1 + fN / 700)
        formant_mel[formant_mel > 0] = (formant_mel[formant_mel > 0] - formant_mel_min) * (
            formant_bin - 2
        ) / (formant_mel_max - formant_mel_min) + 1

        # use 0 or 1
        formant_mel[formant_mel <= 1] = 1
        formant_mel[formant_mel > formant_bin - 1] = formant_bin - 1
        formant_coarse = np.rint(formant_mel).astype(int)
        assert formant_coarse.max() <= formant_bin-1 and formant_coarse.min() >= 1, (
            formant_coarse.max(),
            formant_coarse.min(),
        )
        return formant_coarse

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

    def get_formants(self, x, p_len, formant_shift, formant_to_shift=0, max_number_of_formants=5.5, maximum_formant=5500, pre_emphasis_from=50):
        time_step = self.window / self.sr

        formant = (
            parselmouth.Sound(x, self.sr)
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

        print(f"Shifting formant {formant_to_shift} with a factor of {formant_shift}")
        if formant_to_shift==0:
            formant_shift_diff=f1*(1-formant_shift)
            f1-=formant_shift_diff
            f2-=formant_shift_diff
            f3-=formant_shift_diff
            f4-=formant_shift_diff
            f5-=formant_shift_diff
        elif formant_to_shift==1:
            f1*=formant_shift
        elif formant_to_shift==2:
            f2*=formant_shift
        elif formant_to_shift==3:
            f3*=formant_shift
        elif formant_to_shift==4:
            f4*=formant_shift
        elif formant_to_shift==5:
            f5*=formant_shift

        cf1 = self.coarse_formant(f1)
        cf2 = self.coarse_formant(f2)
        cf3 = self.coarse_formant(f3)
        cf4 = self.coarse_formant(f4)
        cf5 = self.coarse_formant(f5)

        return (f1, f2, f3, f4, f5, cf1, cf2, cf3, cf4, cf5)

    def get_f0(
        self,
        x,
        p_len,
        f0_up_key,
        f0_method,
        crepe_hop_length,
        inp_f0=None,
        input_audio_path=None,
        filter_radius=None
    ):
        global input_audio_path2wav
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        if f0_method == "pm":
            f0 = (
                parselmouth.Sound(x, self.sr)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "harvest":
            input_audio_path2wav[input_audio_path] = x.astype(np.double)
            f0 = cache_harvest_f0(input_audio_path, self.sr, f0_max, f0_min, 10)
            if filter_radius > 2:
                f0 = signal.medfilt(f0, 3)
        elif f0_method == "dio": # Potentially Buggy?
            f0, t = pyworld.dio(
                x.astype(np.double),
                fs=self.sr,
                f0_ceil=f0_max,
                f0_floor=f0_min,
                frame_period=10
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sr)
            f0 = signal.medfilt(f0, 3)
        elif f0_method == "crepe":
            f0 = self.get_f0_official_crepe_computation(x, f0_min, f0_max)
        elif f0_method == "crepe-tiny":
            f0 = self.get_f0_official_crepe_computation(x, f0_min, f0_max, "tiny")
        elif f0_method == "mangio-crepe":
            f0 = self.get_f0_crepe_computation(x, f0_min, f0_max, p_len, crepe_hop_length)
        elif f0_method == "mangio-crepe-tiny":
            f0 = self.get_f0_crepe_computation(x, f0_min, f0_max, p_len, crepe_hop_length, "tiny")
        elif "hybrid" in f0_method:
            # Perform hybrid median pitch estimation
            input_audio_path2wav[input_audio_path] = x.astype(np.double)
            f0 = self.get_f0_hybrid_computation(
                f0_method, 
                input_audio_path,
                x,
                f0_min,
                f0_max,
                p_len,
                filter_radius,
                crepe_hop_length,
                time_step
            )

        f0 *= pow(2, f0_up_key / 12)
        # with open("test.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        tf0 = self.sr // self.window  # 每秒f0点数


        def get_all_notes():
            notes = []
            base_frequency = 440
            octaves_up = 4
            octaves_down = 3

            notes.append(base_frequency)

            for i in range(12*octaves_up):
                notes.append(round(base_frequency*pow(2, (i+1) / 12), 2))

            for i in range(12*octaves_down):
                notes.append(round(base_frequency*pow(2, -(i+1) / 12), 2))

            return notes
        
        def find_closest_note(pitches, notes):
            closest_notes = []
            for pitch in pitches:
                if pitch < 20:
                    closest_note = 0
                else:   
                    closest_note = min(notes, key=lambda x: abs(x - pitch))
                closest_notes.append(closest_note)
            return closest_notes

        import statistics
        
        notes = get_all_notes()
        #steps = 10
        inp_test_f0 = []#[(0, int(p_len*6/17), 155.56), (int(p_len*6/17), int(p_len*9/17), 174.61), (int(p_len*9/17), int(p_len*11.5/17), 277.18), (int(p_len*11.5/17), p_len, 103.83)]#[(i, i+steps, np.random.randint(90, 250)) for i in range(0, f0.shape[0], steps)]#[(0,3604,130.81)]
        for t0, t1, tar_freq in inp_test_f0:
            f0_freq = f0[t0: t1].copy()
            closest_notes = find_closest_note(f0_freq, notes)
            filtered_closest_notes = [n for n in closest_notes if n != 0]
            if not filtered_closest_notes:
                continue
            rep_freq = statistics.mode(filtered_closest_notes)
            idxs = [i for i, x in enumerate(closest_notes) if x == rep_freq]
            mean_diff = rep_freq - f0_freq[idxs].mean()
            diff_freq = tar_freq - rep_freq + mean_diff
            print(f"tar_freq: {tar_freq} | rep_freq: {rep_freq} | diff_freq: {diff_freq} | mean_diff: {mean_diff} | f0_freq: {closest_notes[:10]}")
            new_f0_freq = f0_freq + diff_freq
            f0[t0:t1] = new_f0_freq

        
        if inp_f0 is not None:
            # min_x = inp_f0[:, 0].min()
            # max_x = inp_f0[:, 0].max()
            # delta_t = np.round((max_x - min_x) * tf0 + 1).astype("int16")
            # replace_x = np.linspace(min_x, max_x, delta_t)
            # replace_f0 = np.interp(replace_x, inp_f0[:, 0], inp_f0[:, 1])
            # shape = f0[self.x_pad * tf0: self.x_pad * tf0 + len(replace_f0)].shape[0]
            # f0[self.x_pad * tf0: self.x_pad * tf0 + len(replace_f0)] = replace_f0[:shape]

            delta_t = np.round((inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1).astype("int16")
            replace_f0 = np.interp(list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1])
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[:shape]
        # with open("test_opt.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int)

        return f0_coarse, f0bak  # 1-0

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):  # ,file_index,file_big_npy
        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        t0 = ttime()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
        if protect < 0.5:
            feats0 = feats.clone()
        if (
            isinstance(index, type(None)) == False
            and isinstance(big_npy, type(None)) == False
            and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            # _, I = index.search(npy, 1)
            # npy = big_npy[I.squeeze()]

            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        t1 = ttime()
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch != None and pitchf != None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if protect < 0.5:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            if pitch != None and pitchf != None:
                audio1 = (
                    (net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )
            else:
                audio1 = (
                    (net_g.infer(feats, p_len, sid)[0][0, 0]).data.cpu().float().numpy()
                )
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1

    def get_semb(
        self,
        net_g,
        sid,
    ):
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        with torch.no_grad():
            semb = net_g.infer_sembedding(sid).data.cpu().float().numpy()
        return semb


    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        input_audio_path,
        times,
        f0_up_key,
        f0_method,
        file_index,
        # file_big_npy,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        crepe_hop_length,
        f0_file=None,
    ):
        if (
            file_index != ""
            # and file_big_npy != ""
            # and os.path.exists(file_big_npy) == True
            and os.path.exists(file_index) == True
            and index_rate != 0
        ):
            try:
                index = faiss.read_index(file_index)
                # big_npy = np.load(file_big_npy)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except:
                traceback.print_exc()
                index = big_npy = None
        else:
            index = big_npy = None
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if hasattr(f0_file, "name") == True:
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
            except:
                traceback.print_exc()
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(
                audio_pad,
                p_len,
                f0_up_key,
                f0_method,
                crepe_hop_length,
                inp_f0,
                input_audio_path,
                filter_radius,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if self.device == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        t2 = ttime()
        times[1] += t2 - t1
        for t in opt_ts:
            t = t // self.window * self.window
            if if_f0 == 1:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t
        if if_f0 == 1:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt


    def pipeline2(
        self,
        model,
        net_g,
        audio,
        times,
        f0_up_key,
        f0_method,
        file_index,
        # file_big_npy,
        index_rate,
        if_f0,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        crepe_hop_length,
        f0_file=None,
        sid=None,
        semb=None,
        inter=None,
        function="sid_inf",
        input_audio=None,
        filter_radius=None,
        formant_shift=1,
        formant_to_shift=0,
    ):
        if (
            file_index != ""
            # and file_big_npy != ""
            # and os.path.exists(file_big_npy) == True
            and os.path.exists(file_index) == True
            and index_rate != 0
        ):
            try:
                index = faiss.read_index(file_index)
                # big_npy = np.load(file_big_npy)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except:
                traceback.print_exc()
                index = big_npy = None
        else:
            index = big_npy = None

        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if f0_file:
            try:
                with open(f0_file, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
            except:
                traceback.print_exc()

        if function=="infer_sid":
            sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        if function=="infer_semb":
            semb = torch.tensor(semb, device=self.device).half()
            # if len(semb.shape)<3:
            #     semb = F.normalize(semb.unsqueeze(0)).unsqueeze(-1)
        if function=="infer_inter":
            inter = torch.tensor(inter, device=self.device).half()
        if function=="get_inter":
            sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()


        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(
                audio_pad,
                p_len,
                f0_up_key,
                f0_method,
                crepe_hop_length,
                inp_f0,
                input_audio_path=input_audio,
                filter_radius=filter_radius,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if self.device == "mps":
                pitchf = pitchf.astype(np.float32)
            
            f1, f2, f3, f4, f5, cf1, cf2, cf3, cf4, cf5 = self.get_formants(audio_pad, p_len, formant_shift, formant_to_shift=formant_to_shift)
            f1 = f1[:p_len]
            f2 = f2[:p_len]
            f3 = f3[:p_len]
            f4 = f4[:p_len]
            f5 = f5[:p_len]
            f1[pitchf==0]=0
            f2[pitchf==0]=0
            f3[pitchf==0]=0
            f4[pitchf==0]=0
            f5[pitchf==0]=0
            cf1 = cf1[:p_len]
            cf2 = cf2[:p_len]
            cf3 = cf3[:p_len]
            cf4 = cf4[:p_len]
            cf5 = cf5[:p_len]
            cf1[pitch==1]=1
            cf2[pitch==1]=1
            cf3[pitch==1]=1
            cf4[pitch==1]=1
            cf5[pitch==1]=1

            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
            
            f1 = torch.tensor(f1, device=self.device).unsqueeze(0).half()
            f2 = torch.tensor(f2, device=self.device).unsqueeze(0).half()
            f3 = torch.tensor(f3, device=self.device).unsqueeze(0).half()
            f4 = torch.tensor(f4, device=self.device).unsqueeze(0).half()
            f5 = torch.tensor(f5, device=self.device).unsqueeze(0).float()
            cf1 = torch.tensor(cf1, device=self.device).unsqueeze(0).long()
            cf2 = torch.tensor(cf2, device=self.device).unsqueeze(0).long()
            cf3 = torch.tensor(cf3, device=self.device).unsqueeze(0).long()
            cf4 = torch.tensor(cf4, device=self.device).unsqueeze(0).long()
            cf5 = torch.tensor(cf5, device=self.device).unsqueeze(0).long()

            

        t2 = ttime()
        times[1] += t2 - t1
        if function == "get_inter":
            return self.vc2(
                        model,
                        net_g,
                        sid,
                        semb,
                        inter,
                        audio_pad[t:],
                        pitch[:, t // self.window :] if t is not None else pitch,
                        pitchf[:, t // self.window :] if t is not None else pitchf,
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                        function
                    )
        
        for t in opt_ts:
            t = t // self.window * self.window
            if if_f0 == 1:
                
                audio_opt.append(
                    self.vc2(
                        model,
                        net_g,
                        sid,
                        semb,
                        inter,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                        function,
                        (f1,f2,f3,f4,f5),
                        (cf1,cf2,cf3,cf4,cf5)
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.vc2(
                        model,
                        net_g,
                        sid,
                        semb,
                        inter,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                        function,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t
        if if_f0 == 1:
            audio_opt.append(
                self.vc2(
                    model,
                    net_g,
                    sid,
                    semb,
                    inter,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                    function,
                    (f1,f2,f3,f4,f5),
                    (cf1,cf2,cf3,cf4,cf5)
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self.vc2(
                    model,
                    net_g,
                    sid,
                    semb,
                    inter,
                    audio_pad[t:],
                    None,
                    None,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                    function
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt


    def vc2(
        self,
        model,
        net_g,
        sid,
        semb,
        inter,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        big_npy,
        index_rate,
        version,
        protect,
        function,
        formants,
        coarse_formants
    ):  # ,file_index,file_big_npy
        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        t0 = ttime()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
        if protect < 0.5:
            feats0 = feats.clone()
        if (
            isinstance(index, type(None)) == False
            and isinstance(big_npy, type(None)) == False
            and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            # _, I = index.search(npy, 1)
            # npy = big_npy[I.squeeze()]

            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )
        
        f1, f2, f3, f4, f5 = formants
        cf1, cf2, cf3, cf4, cf5 = coarse_formants

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        t1 = ttime()
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch != None and pitchf != None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]
                f1 = f1[:, :p_len]
                f2 = f2[:, :p_len]
                f3 = f3[:, :p_len]
                f4 = f4[:, :p_len]
                f5 = f5[:, :p_len]
                cf1 = cf1[:, :p_len]
                cf2 = cf2[:, :p_len]
                cf3 = cf3[:, :p_len]
                cf4 = cf4[:, :p_len]
                cf5 = cf5[:, :p_len]

        if protect < 0.5:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len = torch.tensor([p_len], device=self.device).long()

        with torch.no_grad():
            if pitch != None and pitchf != None:
                if function == "infer_sid":
                    audio1 = (
                        (net_g.infer(feats, p_len, pitch, pitchf, cf1, cf2, cf3, cf4, sid)[0][0, 0])
                        .data.cpu()
                        .float()
                        .numpy()
                    )
                elif function == "infer_semb":
                    audio1 = (
                        (net_g.infer_using_sembedding(feats, p_len, pitch, pitchf, cf1, cf2, cf3, cf4, semb)[0][0, 0])
                        .data.cpu()
                        .float()
                        .numpy()
                    )
                elif function == "infer_inter":
                    audio1 = (
                        (net_g.infer_using_inter(inter, pitchf)[0][0, 0])
                        .data.cpu()
                        .float()
                        .numpy()
                    )
                elif function == "get_inter":
                    audio1 = (
                        (net_g.get_inter(feats, p_len, pitch, sid))
                        .data.cpu()
                        .float()
                        .numpy()
                    )
            else:
                if function == "infer_sid":
                    audio1 = (
                        (net_g.infer_using_sembedding(feats, p_len, semb)[0][0, 0]).data.cpu().float().numpy()
                    )
                elif function == "infer_semb":
                    audio1 = (
                        (net_g.infer(feats, p_len, sid)[0][0, 0]).data.cpu().float().numpy()
                    )
                elif function == "infer_inter":
                    pass
                elif function == "get_inter":
                    pass
            
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1