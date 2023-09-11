import sys, os

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))
sys.path.append(os.path.join(now_dir, "train"))
import utils
import datetime

hps = utils.get_hparams()
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
n_gpus = len(hps.gpus.split("-"))
from random import shuffle, randint
import traceback, json, argparse, itertools, math, torch, pdb
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

NUMBER_OF_SPEAKERS = 1#109

#@torch.no_grad()
def get_d_vector_resemblyzer(wav, encoder):
    embeddings = []
    for w in wav:
        embed = encoder.embed_utterance(torch.squeeze(w.cpu()))
        embeddings.append(np.expand_dims(embed,0))
    return torch.tensor(embeddings)

def load_speaker_encoder():
    torch_device = None
    if torch.cuda.is_available():
        torch_device = torch.device(f"cuda:{0 % torch.cuda.device_count()}")
    elif torch.backends.mps.is_available():
        torch_device = torch.device("mps")
    else:
        torch_device = torch.device("cpu")

    encoder = VoiceEncoder(device=torch_device)
    return encoder

speaker_encoder = load_speaker_encoder()
speaker_encoder_function = get_d_vector_resemblyzer

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from infer_pack import commons
from time import sleep
from time import time as ttime
from data_utils import (
    TextAudioLoaderMultiNSFsid,
    TextAudioLoader,
    TextAudioCollateMultiNSFsid,
    TextAudioCollate,
    DistributedBucketSampler,
)

if hps.version == "v1":
    from infer_pack.models import (
        SynthesizerTrnMs256NSFsid as RVC_Model_f0,
        SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
        MultiPeriodDiscriminator,
    )
else:
    from infer_pack.models import (
        SynthesizerTrnMs768NSFsid as RVC_Model_f0,
        SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
        MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
    )
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss, se_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from process_ckpt import savee

global_step = 0


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def main():
    n_gpus = torch.cuda.device_count()
    if torch.cuda.is_available() == False and torch.backends.mps.is_available() == True:
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    children = []
    run(0, n_gpus, hps)
    # for i in range(n_gpus):
    #     subproc = mp.Process(
    #         target=run,
    #         args=(
    #             i,
    #             n_gpus,
    #             hps,
    #         ),
    #     )
    #     children.append(subproc)
    #     subproc.start()

    # for i in range(n_gpus):
    #     children[i].join()


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(
        backend="gloo", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # num_workers=8 -> num_workers=4
    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )

    hps.model.spk_embed_dim = NUMBER_OF_SPEAKERS
    if hps.if_f0 == 1:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
            sr=hps.sample_rate,
            use_d_vectors=hps.use_d_vectors
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
        )
    print("Model hyperparams: ", hps.model)
    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    # net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    # net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    if torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])
    else:
        net_g = DDP(net_g)
        net_d = DDP(net_d)

    try:  # 如果能加载自动resume
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )  # D多半加载没事
        if rank == 0:
            logger.info("loaded D")
        # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        global_step = (epoch_str - 1) * len(train_loader)
        # epoch_str = 1
        # global_step = 0
    except:  # 如果首次不能加载，加载pretrain
        # traceback.print_exc()
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":

            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainG))
            tmp = torch.load(hps.pretrainG, map_location="cpu")["model"]
            if hps.use_d_vectors:
                del tmp['emb_g.weight']
                tmpcond = nn.Conv1d(hps.model.gin_channels, hps.model.upsample_initial_channel, 1)
                tmp['dec.cond.weight'] = tmpcond.weight.data
                tmp['dec.cond.bias'] = tmpcond.bias.data

                for i in range(4):
                    cond_layer = torch.nn.Conv1d(
                        hps.model.gin_channels, 2 * hps.model.hidden_channels * 3, 1
                    )
                    tmp_cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")
                    tmp[f"flow.flows.{i*2}.enc.cond_layer.weight_v"] = tmp_cond_layer.weight_v.data
                    tmp[f"flow.flows.{i*2}.enc.cond_layer.weight_g"] = tmp_cond_layer.weight_g.data
                    tmp[f"flow.flows.{i*2}.enc.cond_layer.bias"] = tmp_cond_layer.bias.data

                # tmp["emb_g.weight"] = nn.Conv1d(256, 256, 1).weight.data
                # tmp["emb_g.bias"] = nn.Conv1d(256, 256, 1).bias.data
                # tmplin = nn.Linear(256, 256)
                # tmp["emb_g.weight"] = tmplin.weight.data
                # tmp["emb_g.bias"] = tmplin.bias.data
            else:
                tmp["emb_g.weight"] = nn.Embedding(NUMBER_OF_SPEAKERS, 256).weight.data

            # Formant Convs for the generator
            # formant_convs = nn.ModuleList()
            # for i, (u, k) in enumerate(zip(hps.model.upsample_rates, hps.model.upsample_kernel_sizes)):
            #     c_cur = hps.model.upsample_initial_channel // (2 ** (i + 1))
            #     if i + 1 < len(hps.model.upsample_rates):
            #         stride_f0 = np.prod(hps.model.upsample_rates[i + 1 :])
            #         formant_convs.append(
            #             nn.Conv1d(
            #                 1,
            #                 c_cur,
            #                 kernel_size=stride_f0 * 2,
            #                 stride=stride_f0,
            #                 padding=stride_f0 // 2,
            #             )
            #         )
            #     else:
            #         formant_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))
            # tmp["dec.formant_convs.0.weight"] = formant_convs[0].weight.data
            # tmp["dec.formant_convs.0.bias"] = formant_convs[0].bias.data
            # tmp["dec.formant_convs.1.weight"] = formant_convs[1].weight.data
            # tmp["dec.formant_convs.1.bias"] = formant_convs[1].bias.data
            # tmp["dec.formant_convs.2.weight"] = formant_convs[2].weight.data
            # tmp["dec.formant_convs.2.bias"] = formant_convs[2].bias.data
            # tmp["dec.formant_convs.3.weight"] = formant_convs[3].weight.data
            # tmp["dec.formant_convs.3.bias"] = formant_convs[3].bias.data

            # Formant hidden layers for the text encoder
            
            # tmplin1 = nn.Linear(1, hps.model.hidden_channels)
            # tmplin2 = nn.Linear(1, hps.model.hidden_channels)
            # tmplin3 = nn.Linear(1, hps.model.hidden_channels)
            # tmplin4 = nn.Linear(1, hps.model.hidden_channels)
            # tmp["enc_p.emb_formant1.weight"] = tmplin1.weight.data
            # tmp["enc_p.emb_formant1.bias"] = tmplin1.bias.data
            # tmp["enc_p.emb_formant2.weight"] = tmplin2.weight.data
            # tmp["enc_p.emb_formant2.bias"] = tmplin2.bias.data
            # tmp["enc_p.emb_formant3.weight"] = tmplin3.weight.data
            # tmp["enc_p.emb_formant3.bias"] = tmplin3.bias.data
            # tmp["enc_p.emb_formant4.weight"] = tmplin4.weight.data
            # tmp["enc_p.emb_formant4.bias"] = tmplin4.bias.data

            #TODO: USE THESE FOR FORMANTS
            # tmp["enc_p.emb_formant1.weight"] = nn.Embedding(256, hps.model.hidden_channels).weight.data
            # tmp["enc_p.emb_formant2.weight"] = nn.Embedding(256, hps.model.hidden_channels).weight.data
            # tmp["enc_p.emb_formant3.weight"] = nn.Embedding(256, hps.model.hidden_channels).weight.data
            # tmp["enc_p.emb_formant4.weight"] = nn.Embedding(256, hps.model.hidden_channels).weight.data

            # tmp["enc_p.emb_formant5.weight"] = nn.Embedding(256, hps.model.hidden_channels).weight.data

            # tmp["dec.emb_formant1.weight"] = nn.Embedding(256, hps.model.upsample_initial_channel).weight.data
            # tmp["dec.emb_formant2.weight"] = nn.Embedding(256, hps.model.upsample_initial_channel).weight.data
            # tmp["dec.emb_formant3.weight"] = nn.Embedding(256, hps.model.upsample_initial_channel).weight.data
            # tmp["dec.emb_formant4.weight"] = nn.Embedding(256, hps.model.upsample_initial_channel).weight.data

            # tmp["emb_formant1.weight"] = nn.Embedding(256, hps.model.gin_channels).weight.data
            # tmp["emb_formant2.weight"] = nn.Embedding(256, hps.model.gin_channels).weight.data
            # tmp["emb_formant3.weight"] = nn.Embedding(256, hps.model.gin_channels).weight.data

            print(
                net_g.module.load_state_dict(
                    tmp
                )
            )  ##测试不加载优化器
        if hps.pretrainD != "":

            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainD))
            print(
                net_d.module.load_state_dict(
                    torch.load(hps.pretrainD, map_location="cpu")["model"]
                )
            )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    cache = []
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer, writer_eval],
                cache,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
                cache,
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, cache
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    # Prepare data iterator
    if hps.if_cache_data_in_gpu == True:
        # Use Cache
        data_iterator = cache
        if cache == []:
            # Make new cache
            for batch_idx, info in enumerate(train_loader):
                # Unpack
                if hps.if_f0 == 1:
                    (
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        f1,
                        f2,
                        f3,
                        f4,
                        f5,
                        cf1,
                        cf2,
                        cf3,
                        cf4,
                        cf5,
                        d_vector,
                        sid,
                    ) = info
                else:
                    (
                        phone,
                        phone_lengths,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                # Load on CUDA
                if torch.cuda.is_available():
                    phone = phone.cuda(rank, non_blocking=True)
                    phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                    if hps.if_f0 == 1:
                        pitch = pitch.cuda(rank, non_blocking=True)
                        pitchf = pitchf.cuda(rank, non_blocking=True)
                        f1 = f1.cuda(rank, non_blocking=True)
                        f2 = f2.cuda(rank, non_blocking=True)
                        f3 = f3.cuda(rank, non_blocking=True)
                        f4 = f4.cuda(rank, non_blocking=True)
                        f5 = f5.cuda(rank, non_blocking=True)
                        cf1 = cf1.cuda(rank, non_blocking=True)
                        cf2 = cf2.cuda(rank, non_blocking=True)
                        cf3 = cf3.cuda(rank, non_blocking=True)
                        cf4 = cf4.cuda(rank, non_blocking=True)
                        cf5 = cf5.cuda(rank, non_blocking=True)
                    d_vector = d_vector.cuda(rank, non_blocking=True)
                    sid = sid.cuda(rank, non_blocking=True)
                    spec = spec.cuda(rank, non_blocking=True)
                    spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                    wave = wave.cuda(rank, non_blocking=True)
                    wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
                # Cache on list
                if hps.if_f0 == 1:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                pitch,
                                pitchf,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                f1,
                                f2,
                                f3,
                                f4,
                                f5,
                                cf1,
                                cf2,
                                cf3,
                                cf4,
                                cf5,
                                d_vector,
                                sid,
                            ),
                        )
                    )
                else:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
        else:
            # Load shuffled cache
            shuffle(cache)
    else:
        # Loader
        data_iterator = enumerate(train_loader)

    # Run steps
    epoch_recorder = EpochRecorder()
    for batch_idx, info in data_iterator:
        # Data
        ## Unpack
        if hps.if_f0 == 1:
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                f1,
                f2,
                f3,
                f4,
                f5,
                cf1,
                cf2,
                cf3,
                cf4,
                cf5,
                d_vector,
                sid,
            ) = info
        else:
            phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
        ## Load on CUDA
        if (hps.if_cache_data_in_gpu == False) and torch.cuda.is_available():
            phone = phone.cuda(rank, non_blocking=True)
            phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
            if hps.if_f0 == 1:
                pitch = pitch.cuda(rank, non_blocking=True)
                pitchf = pitchf.cuda(rank, non_blocking=True)
                f1 = f1.cuda(rank, non_blocking=True)
                f2 = f2.cuda(rank, non_blocking=True)
                f3 = f3.cuda(rank, non_blocking=True)
                f4 = f4.cuda(rank, non_blocking=True)
                f5 = f5.cuda(rank, non_blocking=True)
                cf1 = cf1.cuda(rank, non_blocking=True)
                cf2 = cf2.cuda(rank, non_blocking=True)
                cf3 = cf3.cuda(rank, non_blocking=True)
                cf4 = cf4.cuda(rank, non_blocking=True)
                cf5 = cf5.cuda(rank, non_blocking=True)
            d_vector = d_vector.cuda(rank, non_blocking=True)
            sid = sid.cuda(rank, non_blocking=True)
            spec = spec.cuda(rank, non_blocking=True)
            spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
            wave = wave.cuda(rank, non_blocking=True)
            # wave_lengths = wave_lengths.cuda(rank, non_blocking=True)

        # Calculate
        with autocast(enabled=hps.train.fp16_run):
            if hps.if_f0 == 1:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, aux_input={"d_vectors": d_vector, "speaker_ids": sid}) #cf1, cf2, cf3, cf4,
            else:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            if hps.train.fp16_run == True:
                y_hat_mel = y_hat_mel.half()
            wave = commons.slice_segments(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)
        spk_encoder_loss_alpha = 9.0
        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
                if hps.use_se_loss:
                    loss_se = se_loss(wave, y_hat, speaker_encoder_function, speaker_encoder)*spk_encoder_loss_alpha
                    loss_gen_all = loss_gen_all + loss_se
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                # Amor For Tensorboard display
                if loss_mel > 75:
                    loss_mel = 75
                if loss_kl > 9:
                    loss_kl = 9

                logger.info([global_step, lr])
                logger.info(
                    f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f},loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}"
                )
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/kl": loss_kl,
                    }
                )

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )
        global_step += 1
    # /Run steps

    if epoch % hps.save_every_epoch == 0 and rank == 0:
        if hps.if_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
            )
        else:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(2333333)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(2333333)),
            )
        if rank == 0 and hps.save_every_weights == "1":
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            logger.info(
                "saving ckpt %s_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    savee(
                        ckpt,
                        hps.sample_rate,
                        hps.if_f0,
                        hps.name + "_e%s_s%s" % (epoch, global_step),
                        epoch,
                        hps.version,
                        hps,
                    ),
                )
            )

    if rank == 0:
        logger.info("====> Epoch: {} {}".format(epoch, epoch_recorder.record()))
    if epoch >= hps.total_epoch and rank == 0:
        logger.info("Training is done. The program is closed.")

        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        logger.info(
            "saving final ckpt:%s"
            % (
                savee(
                    ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps.version, hps
                )
            )
        )
        sleep(1)
        os._exit(2333333)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
