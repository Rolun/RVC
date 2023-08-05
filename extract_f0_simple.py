import torch
from torch import Tensor
import numpy as np
import parselmouth
import pyworld
import torchcrepe

class FeatureInput(object):
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0_from_audio(self, audio, f0_method, crepe_hop_length):
        x = audio
        p_len = x.shape[0] // self.hop
        if f0_method == "pm":
            time_step = 160 / 16000 * 1000
            f0 = (
                parselmouth.Sound(x, self.fs)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=self.f0_min,
                    pitch_ceiling=self.f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "harvest":
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        elif f0_method == "dio":
            f0, t = pyworld.dio(
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        elif f0_method == "crepe": # Fork Feature: Added crepe f0 for f0 feature extraction
            # Pick a batch size that doesn't cause memory errors on your gpu
            torch_device_index = 0
            torch_device = None
            if torch.cuda.is_available():
                torch_device = torch.device(f"cuda:{torch_device_index % torch.cuda.device_count()}")
            elif torch.backends.mps.is_available():
                torch_device = torch.device("mps")
            else:
                torch_device = torch.device("cpu")
            model = "full"
            batch_size = 512
            # Compute pitch using first gpu
            audio = torch.tensor(np.copy(x))[None].float()
            f0, pd = torchcrepe.predict(
                audio,
                self.fs,
                160,
                self.f0_min,
                self.f0_max,
                model,
                batch_size=batch_size,
                device=torch_device,
                return_periodicity=True,
            )
            pd = torchcrepe.filter.median(pd, 3)
            f0 = torchcrepe.filter.mean(f0, 3)
            f0[pd < 0.1] = 0
            f0 = f0[0].cpu().numpy()
        elif f0_method == "mangio-crepe":
            print("Performing crepe pitch extraction. (EXPERIMENTAL)")
            print("CREPE PITCH EXTRACTION HOP LENGTH: " + str(crepe_hop_length))
            x = x.astype(np.float32)
            x /= np.quantile(np.abs(x), 0.999)
            torch_device_index = 0
            torch_device = None
            if torch.cuda.is_available():
                torch_device = torch.device(f"cuda:{torch_device_index % torch.cuda.device_count()}")
            elif torch.backends.mps.is_available():
                torch_device = torch.device("mps")
            else:
                torch_device = torch.device("cpu")
            audio = torch.from_numpy(x).to(torch_device, copy=True)
            audio = torch.unsqueeze(audio, dim=0)
            if audio.ndim == 2 and audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True).detach()
            audio = audio.detach()
            print(
                "Initiating f0 Crepe Feature Extraction with an extraction_crepe_hop_length of: " +
                str(crepe_hop_length)
            )
            # Pitch prediction for pitch extraction
            pitch: Tensor = torchcrepe.predict(
                audio,
                self.fs,
                crepe_hop_length,
                self.f0_min,
                self.f0_max,
                "full",
                batch_size=crepe_hop_length * 2,
                device=torch_device,
                pad=True                
            )
            p_len = p_len or x.shape[0] // crepe_hop_length
            # Resize the pitch
            source = np.array(pitch.squeeze(0).cpu().float().numpy())
            source[source < 0.001] = np.nan
            target = np.interp(
                np.arange(0, len(source) * p_len, len(source)) / p_len,
                np.arange(0, len(source)),
                source
            )
            f0 = np.nan_to_num(target)
        elif "hybrid" in f0_method: # EXPERIMENTAL
            # Perform hybrid median pitch estimation
            time_step = 160 / 16000 * 1000
            f0 = self.get_f0_hybrid_computation(
                f0_method, 
                x,
                self.f0_min,
                self.f0_max,
                p_len,
                crepe_hop_length,
                time_step
            )
        # Mangio-RVC-Fork Feature: Add hybrid f0 inference to feature extraction. EXPERIMENTAL...

        return f0