import os,sys
now_dir = os.getcwd()
sys.path.append(now_dir)
print(sys.argv)
import sys
import torch
from multiprocessing import cpu_count
import numpy as np
import time
import json

class Config:
    def __init__(self,device,is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("16系/10系显卡和P40强制单精度")
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    with open(f"configs/{config_file}", "r") as f:
                        strr = f.read().replace("true", "false")
                    with open(f"configs/{config_file}", "w") as f:
                        f.write(strr)
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            print("没有发现支持的N卡, 使用MPS进行推理")
            self.device = "mps"
        else:
            print("没有发现支持的N卡, 使用CPU进行推理")
            self.device = "cpu"
            self.is_half = True

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max


from vc_infer_pipeline import VC
from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from my_utils import load_audio
from fairseq import checkpoint_utils
from scipy.io import wavfile

def load_hubert(device, is_half):
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"],suffix="",)
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if(is_half):hubert_model = hubert_model.half()
    else:hubert_model = hubert_model.float()
    hubert_model.eval()
    return hubert_model

def vc_single(input_audio, f0_up_key, f0_file, f0_method, file_index, index_rate, filter_radius, resample_sr, rms_mix_rate, protect, tgt_sr, net_g, vc, hubert_model, version, cpt, crepe_hop_length, sid = None, semb = None, inter=None, function="infer_sid", formant_shift=1):
    if input_audio is None:return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    audio=load_audio(input_audio,16000)
    times = [0, 0, 0]
    if(hubert_model==None):load_hubert()
    if_f0 = cpt.get("f0", 1)

    if function == "infer_d_vector":
        # d_vector = get_d_vector_yourtts(input_audio, se_model_path, se_config_path)
        d_vector = get_d_vector_resemblyzer(input_audio)
        function = "infer_semb"
        semb = d_vector
        
    # audio_opt=vc.pipeline(hubert_model,net_g,sid,audio,times,f0_up_key,f0_method,file_index,file_big_npy,index_rate,if_f0,f0_file=f0_file)
    audio_opt=vc.pipeline2(
        model = hubert_model,
        net_g = net_g,
        audio = audio,
        times = times,
        f0_up_key = f0_up_key,
        f0_method = f0_method,
        file_index = file_index,
        index_rate = index_rate,
        if_f0 = if_f0,
        tgt_sr = tgt_sr,
        resample_sr = resample_sr,
        rms_mix_rate = rms_mix_rate,
        version = version,
        protect = protect,
        crepe_hop_length = crepe_hop_length,
        f0_file=f0_file,
        sid=sid,
        semb=semb,
        inter=inter,
        function=function,
        input_audio=input_audio,
        filter_radius=filter_radius,
        formant_shift=formant_shift
    )
    print(times)
    return audio_opt


def get_vc(model_path, device_config, is_half, use_d_vector = False):
    print("loading pth %s"%model_path)
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"] = list(cpt["config"])
    # if not use_d_vector:
    cpt["config"][-3]=cpt["weight"]["emb_g.weight"].shape[0]#n_spk
    if_f0=cpt.get("f0",1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:#
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], use_d_vectors=use_d_vector, is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print("Load state: ", net_g.load_state_dict(cpt["weight"], strict=False))  # 不加这一行清不干净，真奇葩
    net_g.eval().to(device_config.device)
    if (device_config.is_half):net_g = net_g.half()
    else:net_g = net_g.float()
    vc = VC(tgt_sr, device_config)
    n_spk=cpt["config"][-3]
    return n_spk, tgt_sr, vc, cpt, version, net_g
    # return {"visible": True,"maximum": n_spk, "__type__": "update"}

device = "cuda:0"
is_half = True
model_path = "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/weights/NUSR8E-formant-experiment-all-encoder_e250_s23500.pth" #merged3_e185_s8880.pth
input_path = "C:/Users/lundb/Documents/Other/Music/datasets/clean_singer/JLEE/08.wav"#"C:/Users/lundb/Documents/Other/Music/i.wav"#"C:/Users/lundb/Documents/Other/Music/Martin_recordings/martin_hq.wav"
f0method = "mangio-crepe"
index_path = ""#"C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/logs/sandro_sid/added_IVF777_Flat_nprobe_1_sandro_sid_v2.index"
index_rate = 0.7
filter_radius = 3
resample_sr = 0
rms_mix_rate = 0.2
protect = 0.33
crepe_hop_length = 64
f0_file = None#"C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/test_input_f0.txt"
se_model_path = "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/speaker_embeddings/model_se.pth"
se_config_path = "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/speaker_embeddings/config_se.json" 
use_d_vector = False


def get_semb(sid, output_path = ""):
    device_config = Config(device, is_half)

    n_spk, tgt_sr, vc, cpt, version, net_g = get_vc(model_path, device_config, is_half, use_d_vector)
    semb = vc.get_semb(net_g, sid)
    if output_path:
        np.save(output_path, semb)
    print("semb saved")
    return semb

def get_inter(sid, f0up_key=0):
    print("Started")
    device_config = Config(device, is_half)

    n_spk, tgt_sr, vc, cpt, version, net_g = get_vc(model_path, device_config, is_half, use_d_vector)
    hubert_model = load_hubert(device, is_half)
    inter = vc_single(input_path,f0up_key,None,f0method,index_path,index_rate,filter_radius,resample_sr,rms_mix_rate,protect,tgt_sr,net_g,vc,hubert_model,version,cpt,crepe_hop_length,sid=sid,function="get_inter")
    np.save(str(sid)+"_inter.npy", inter)

def get_d_vector_yourtts(audio_file, model_path, config_path, old_speakers_file = None, use_cuda = None):
    from TTS.tts.utils.speakers import SpeakerManager
    use_cuda = torch.cuda.is_available() if use_cuda==None else use_cuda

    encoder_manager = SpeakerManager(
        encoder_model_path=model_path,
        encoder_config_path=config_path,
        d_vectors_file_path=old_speakers_file,
        use_cuda=use_cuda,
    )
    d_vector = encoder_manager.compute_embedding_from_clip(audio_file)
    return d_vector

def get_d_vector_resemblyzer(audio_file, save_path=""):
    from resemblyzer import VoiceEncoder, preprocess_wav
    from pathlib import Path
    import numpy as np

    fpath = Path(audio_file)
    wav = preprocess_wav(fpath)

    encoder = VoiceEncoder()
    embed = encoder.embed_utterance(wav)
    np.set_printoptions(precision=3, suppress=True)
    if save_path:
        np.save(save_path, embed)
    return embed

# def get_d_vector_ge2e(audio_file):
#     from speaker_embeddings.extract_ge2e_embeddings import load_model, extract_embed_utterance
#     model = load_model("C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/speaker_embeddings/pretrained_GE2E/encoder/saved_models/pretrained.pt")
#     embed = extract_embed_utterance(model, )

def generate(inp, function="infer_sid", f0up_key = 0, formant_shift=1, output_path = "test.wav"):
    print("Started")
    device_config = Config(device, is_half)

    n_spk, tgt_sr, vc, cpt, version, net_g = get_vc(model_path, device_config, is_half, use_d_vector)
    hubert_model = load_hubert(device, is_half)
    if function == "infer_sid":
        wav_opt=vc_single(input_path,f0up_key,None,f0method,index_path,index_rate,filter_radius,resample_sr,rms_mix_rate,protect,tgt_sr,net_g,vc,hubert_model,version,cpt,crepe_hop_length,sid=inp,function="infer_sid", formant_shift=formant_shift)
    if function == "infer_semb":
        wav_opt=vc_single(input_path,f0up_key,None,f0method,index_path,index_rate,filter_radius,resample_sr,rms_mix_rate,protect,tgt_sr,net_g,vc,hubert_model,version,cpt,crepe_hop_length,semb=inp,function="infer_semb")
    if function == "infer_inter":
        wav_opt=vc_single(input_path,f0up_key,f0_file,f0method,index_path,index_rate,filter_radius,resample_sr,rms_mix_rate,protect,tgt_sr,net_g,vc,hubert_model,version,cpt,crepe_hop_length,inter=inp,function="infer_inter")
    if function == "infer_d_vector":
        wav_opt=vc_single(input_path,f0up_key,None,f0method,index_path,index_rate,filter_radius,resample_sr,rms_mix_rate,protect,tgt_sr,net_g,vc,hubert_model,version,cpt,crepe_hop_length,function="infer_d_vector")
    # import pdb; pdb.set_trace()
    wavfile.write(output_path, tgt_sr, wav_opt)



def semb_interpol(semb1, semb2, step, steps):
    delta = (semb2-semb1)/steps
    return delta*step + semb1


def create_embedding_mapping(max_sid, output_path = "embedding_mapping.json"):
    mapping = {}
    for sid in range(max_sid):
        semb = get_semb(sid)
        mapping[sid] = semb.tolist()

    with open(output_path, 'w') as f: 
        json.dump(mapping, f)


def create_f0_mapping(logs_path, output_path = "average_pitch_mapping.json"):
    initial_mapping = {}
    files = os.listdir(logs_path)
    for file in files:
        sid = file.split("_")[0]
        if sid not in initial_mapping:
            initial_mapping[sid] = []
        loaded_file = np.load(os.path.join(logs_path, file))
        initial_mapping[sid].extend(loaded_file)

    mapping = {}
    for key, value in initial_mapping.items():
        sorted_f0 = np.sort(value)
        mapping[key] = np.median(sorted_f0)

    with open(output_path, 'w') as f: 
        json.dump(mapping, f)

# get_inter(sid=0,f0up_key=0)
# get_inter(sid=1,f0up_key=0)
# get_inter(sid=2,f0up_key=6)
# get_inter(sid=3,f0up_key=8)

# inter0 = np.load("0_inter.npy")
# inter1 = np.load("1_inter.npy")
# inter2 = np.load("2_inter.npy")
# inter3 = np.load("3_inter.npy")
# generate_with_inter(inter0)

# steps = 5
# fup_key_list = [0,1,3,5,6]
# for i in range(steps):
#     inter = semb_interpol(inter1, inter3, i+1, steps)
#     generate_with_inter(inter, f0up_key = 0, savename="test_"+str(i)+".wav")

# for i in range(1,109):
#     get_semb(i, output_path=f"embeddings/{i}_emb.npy")

# semb0 = np.load("d_vector_tmp_0.npy")
# semb1 = np.load("d_vector_tmp_1.npy")
# semb2 = np.load("d_vector_tmp_2.npy")
# semb3 = np.load("d_vector_tmp_3.npy")
# semb0 = np.load("0_embeddings.npy")
# semb1 = np.load("1_embeddings.npy")
# semb2 = np.load("2_embeddings.npy")
# semb3 = np.load("3_embeddings.npy")
# generate(semb0, function="infer_semb", output_path="test_stuff/test_0.wav")
# generate(semb1, function="infer_semb", output_path="test_stuff/test_1.wav")
# generate(semb2, function="infer_semb", output_path="test_stuff/test_2.wav")
# generate(semb3, function="infer_semb", output_path="test_stuff/test_3.wav")

# male = (semb0 + semb1) / 2
# female = (semb2 + semb3) / 2
# test = semb2-female+male
# generate_with_semb(test, f0up_key = 0, savename="test1"+".wav")
# generate_with_semb(semb2, f0up_key = 0, savename="test2"+".wav")

# steps = 5
# for i in range(steps):
#     semb = semb_interpol(semb1, semb3, i+1, steps)
#     generate_with_semb(semb, f0up_key = 0, savename="test_"+str(i)+".wav")

# semb = np.zeros((1,256,1))
# semb = np.random.rand(1,256,1)
# np.random.shuffle(semb.flat)

# get_d_vector_resemblyzer("C:/Users/lundb/Documents/Other/Music/Voices/Multi-speaker-training/aloe_blacc-refined/Aloe-Blacc-I_Need_A_Dollar.flac", save_path="C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/d_vector_1.npy")
# get_d_vector_resemblyzer("C:/Users/lundb/Documents/Other/Music/Voices/Multi-speaker-training/rita-refined/Your_Song_ft._Ed_Sheeran.flac", save_path="C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/d_vector_2.npy")
# get_d_vector_resemblyzer("C:/Users/lundb/Documents/Other/Music/Voices/Multi-speaker-training/sandro_multiple_refined/Track 1_refined.flac", save_path="C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/d_vector_3.npy")
# get_d_vector_resemblyzer("C:/Users/lundb/Documents/Other/Music/Voices/Multi-speaker-training/whitney-refined/How_Will_I_Know.flac", save_path="C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/d_vector_4.npy")
# get_d_vector_resemblyzer("C:/Users/lundb/Documents/Other/Music/Voices/Multi-speaker-training/zara-refined/End_Of_Time.flac", save_path="C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/d_vector_5.npy")


# d_vector1 = np.load("d_vector_1.npy")
# d_vector2 = np.load("d_vector_2.npy")
# d_vector3 = np.load("d_vector_3.npy")
# d_vector4 = np.load("d_vector_4.npy")
# d_vector5 = np.load("d_vector_5.npy")

# for i in range(0, 14):
#     generate(i, function="infer_sid", output_path=f"test_stuff/mixed_{i}.wav")
# generate(0, function="infer_sid", f0up_key=7, output_path="test_stuff/test2.wav")
generate(0, function="infer_sid", f0up_key=-8, formant_shift=0.8, output_path="test_stuff/formant_test_encoder_all_f-8_0.8.wav")
# generate(2, function="infer_sid", output_path="test_stuff/test_2.wav")
# generate(3, function="infer_sid", output_path="test_stuff/test_3.wav")
# generate(None, function="infer_d_vector")
# generate(d_vector1, function="infer_semb", output_path="test_stuff/test_1.wav")
# generate(d_vector2, function="infer_semb", output_path="test_stuff/test_2.wav")
# generate(d_vector3, function="infer_semb", output_path="test_stuff/test_3.wav")
# generate(d_vector4, function="infer_semb", output_path="test_stuff/test_4.wav")
# generate(d_vector5, function="infer_semb", output_path="test_stuff/test_5.wav")


# o_semb = np.load("embeddings/1_emb.npy")
# gen_semb = np.expand_dims(np.load("VAE/generated_emb_1.npy"),-1)

# generate(o_semb, function="infer_semb", output_path="test_stuff/test_0.wav")
# generate(gen_semb, function="infer_semb", output_path="test_stuff/test_1.wav")


# create_embedding_mapping(14)
# create_f0_mapping("logs/mixed_dataset_test_freq/2b-f0nsf")