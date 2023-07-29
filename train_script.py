import torch, os, traceback, sys, warnings, shutil, numpy as np

os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
import threading
from time import sleep
from subprocess import Popen
import faiss
from random import shuffle
import json

now_dir = os.getcwd()
sys.path.append(now_dir)
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
from i18n import I18nAuto
import ffmpeg
from MDXNet import MDXNetDereverb

i18n = I18nAuto()
i18n.print()
# 判断是否有能用来训练和加速推理的N卡
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if (not torch.cuda.is_available()) or ngpu == 0:
    if_gpu_ok = False
else:
    if_gpu_ok = False
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if (
            "10" in gpu_name
            or "16" in gpu_name
            or "20" in gpu_name
            or "30" in gpu_name
            or "40" in gpu_name
            or "A2" in gpu_name.upper()
            or "A3" in gpu_name.upper()
            or "A4" in gpu_name.upper()
            or "P4" in gpu_name.upper()
            or "A50" in gpu_name.upper()
            or "A60" in gpu_name.upper()
            or "70" in gpu_name
            or "80" in gpu_name
            or "90" in gpu_name
            or "M4" in gpu_name.upper()
            or "T4" in gpu_name.upper()
            or "TITAN" in gpu_name.upper()
        ):  # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok == True and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])
from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
import soundfile as sf
from fairseq import checkpoint_utils
import gradio as gr
import logging
from vc_infer_pipeline import VC
from config import Config
from infer_uvr5 import _audio_pre_, _audio_pre_new
from my_utils import load_audio
from train.process_ckpt import show_info, change_info, merge, extract_small_model

config = Config()
# from trainset_preprocess_pipeline import PreProcess
logging.getLogger("numba").setLevel(logging.WARNING)


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


hubert_model = None

def if_done(done, p):
    while 1:
        if p.poll() == None:
            sleep(0.5)
        else:
            break
    done[0] = True

def if_done_multi(done, ps):
    while 1:
        # poll==None代表进程未结束
        # 只要有一个进程未结束都不停
        flag = 1
        for p in ps:
            if p.poll() == None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True

sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    cmd = (
        config.python_cmd
        + " trainset_preprocess_pipeline_print_multi_folders.py %s %s %s %s/logs/%s "
        % (trainset_dir, sr, n_p, now_dir, exp_dir)
        + str(config.noparallel)
    )
    print(cmd)
    p = Popen(cmd, shell=True)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE,cwd=now_dir
    done = [False]
    threading.Thread(
        target=if_done,
        args=(done,p,),
    ).start()

    while not done[0]:
        sleep(0.5)

def extract_feature(gpus, n_p, f0method, if_f0, use_d_vectors, exp_dir, version19, echl):
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        cmd = config.python_cmd + " extract_f0_print.py %s/logs/%s %s %s %s" % (
            now_dir,
            exp_dir,
            n_p,
            f0method,
            echl,
        )
        print(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
        ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
        done1 = [False]
        threading.Thread(
            target=if_done,
            args=(
                done1,
                p,
            ),
        ).start()
    else:
        done1 = [True]

    if use_d_vectors:
        cmd = config.python_cmd + " create_d_vectors.py %s/logs/%s" % (
            now_dir,
            exp_dir,
        )
        print(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
        ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
        done2 = [False]
        threading.Thread(
            target=if_done,
            args=(
                done2,
                p,
            ),
        ).start()
    else:
        done2 = [True]

        
    ####对不同part分别开多进程
    """
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    """
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            config.python_cmd
            + " extract_feature_print.py %s %s %s %s %s/logs/%s %s"
            % (
                config.device,
                leng,
                idx,
                n_g,
                now_dir,
                exp_dir,
                version19,
            )
        )
        print(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done3 = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done3,
            ps,
        ),
    ).start()

    while not (done1[0] and done2[0] and done3[0]):
        sleep(0.5)

def train_index(exp_dir1, version19):
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if os.path.exists(feature_dir) == False:
        return "请先进行特征提取!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "请先进行特征提取！"
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    # n_ivf =  big_npy.shape[0] // 39
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos = []
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    # faiss.write_index(index, '%s/trained_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append(
        "成功构建索引，added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    yield "\n".join(infos)

def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    use_d_vectors,
    use_se_loss,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    # 生成filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    if use_d_vectors:
        d_vector_dir = "%s/4_d_vectors" % (exp_dir)
        names = set(names) & set([name.split(".")[0] for name in os.listdir(d_vector_dir)])

    opt = []
    spk_list = []
    for name in names:
        spk_id5 = name.split("_")[0]
        if spk_id5 not in spk_list:
            spk_list.append(spk_id5)
        if if_f0_3:
            if use_d_vectors:
                opt.append(
                    "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s/%s.npy|%s"
                    % (
                        gt_wavs_dir.replace("\\", "\\\\"),
                        name,
                        feature_dir.replace("\\", "\\\\"),
                        name,
                        f0_dir.replace("\\", "\\\\"),
                        name,
                        f0nsf_dir.replace("\\", "\\\\"),
                        name,
                        d_vector_dir.replace("\\", "\\\\"),
                        name,
                        spk_id5,
                    )
                )
            else:
                opt.append(
                    "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                    % (
                        gt_wavs_dir.replace("\\", "\\\\"),
                        name,
                        feature_dir.replace("\\", "\\\\"),
                        name,
                        f0_dir.replace("\\", "\\\\"),
                        name,
                        f0nsf_dir.replace("\\", "\\\\"),
                        name,
                        spk_id5,
                    )
                )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        if use_d_vectors:
            for _ in range(2):
                opt.append(
                    "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s/logs/mute/4_d_vectors/mute.npy|%s"
                    % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, now_dir, spk_id5)
                )
        else:
            for spk_id5 in spk_list:
                for _ in range(2):
                    opt.append(
                        "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                        % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
                    )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    print("write filelist done")

    # 生成config#无需生成config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    print("use gpus:", gpus16)
    if pretrained_G14 == "":
        print("no pretrained Generator")
    if pretrained_D15 == "":
        print("no pretrained Discriminator")
    if gpus16:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s -dv %s -sel %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                ("-pg %s" % pretrained_G14) if pretrained_G14 != "" else "",
                ("-pd %s" % pretrained_D15) if pretrained_D15 != "" else "",
                1 if if_save_latest13 == True else 0,
                1 if if_cache_gpu17 == True else 0,
                1 if if_save_every_weights18 == True else 0,
                version19,
                1 if use_d_vectors else 0,
                1 if use_se_loss else 0,
            )
        )
    else:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                ("-pg %s" % pretrained_G14) if pretrained_G14 != "" else "\b",
                ("-pd %s" % pretrained_D15) if pretrained_D15 != "" else "\b",
                1 if if_save_latest13 == True else 0,
                1 if if_cache_gpu17 == True else 0,
                1 if if_save_every_weights18 == True else 0,
                version19,
            )
        )
    print(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return "训练结束, 您可查看控制台训练日志或实验文件夹下的train.log"

def main():
    use_d_vectors = False
    use_se_loss = False
    trainset_dir4 = "C:/Users/lundb/Documents/Other/Music/datasets/VocalSet11Small" #Trainingset folder
    exp_dir1 = "vocalset" #Experiment name
    sr2 = "40k" #Target sample rate
    if_f0_3 = True #Pitch guidance, required for singing
    # trainset_dir4 = "Training" #Training data folder
    np7 = 2 #Parallell processes
    f0method8 = "mangio-crepe" #Pitch extraction method
    save_epoch10 = 5 #Save frequency
    total_epoch11 = 250 #Total epochs
    batch_size12 = 16 #Batch size
    if_save_latest13 = True #Save only the latest .ckpt file
    pretrained_G14 = "pretrained_v2/f0G40k.pth"
    pretrained_D15 = "pretrained_v2/f0D40k.pth"
    gpus6 = gpus16 = "0" #Enter the GPU index(es) separated by '-', e.g., 0-1-2 to use GPU 0, 1, and 2
    if_cache_gpu17 = False #Cache all training sets to GPU memory
    if_save_every_weights18 = True #Save a small final model to the 'weights' folder at each save point
    version19 = "v2" #v1 or v2
    extraction_crepe_hop_length = 64

    # preprocess_dataset(
    #     trainset_dir4, 
    #     exp_dir1,
    #     sr2, 
    #     np7
    # )

    # print("preprocess done")

    extract_feature(gpus6, np7, f0method8, if_f0_3, use_d_vectors, exp_dir1, version19, extraction_crepe_hop_length)

    print("feature extraction done")

    train_index(exp_dir1, version19)

    print("index trained")

    click_train(
        exp_dir1,
        sr2,
        if_f0_3,
        use_d_vectors,
        use_se_loss,
        save_epoch10,
        total_epoch11,
        batch_size12,
        if_save_latest13,
        pretrained_G14,
        pretrained_D15,
        gpus16,
        if_cache_gpu17,
        if_save_every_weights18,
        version19
    )

main()