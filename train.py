from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler  import OneCycleLR
from torch import nn
import torch
import numpy as np

from tqdm import tqdm
import argparse
import wandb
import os
import sys
sys.path.append('src/')

from config import train_config, model_config
from dataloader import get_data_to_buffer, BufferDataset, collate_fn_tensor
from wandb_writer import WanDBWriter
from FastSpeech2 import FastSpeech
import waveglow
import text
import utils


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration_predicted, pitch_prediction, energy_prediction, mel_target, duration_predictor_target, pitch_target, energy_target):
        mel_loss = self.l1_loss(mel, mel_target)

        duration_predictor_loss = self.mse_loss(duration_predicted,
                                               torch.log1p(duration_predictor_target.float()))
        pitch_loss = self.mse_loss(pitch_prediction, torch.log1p(pitch_target))
        energy_loss = self.mse_loss(energy_prediction, torch.log1p(energy_target))
        return mel_loss, duration_predictor_loss, pitch_loss, energy_loss

def synthesis(model, text, alpha=1.0, p_alpha=1.0, e_alpha=1.0):
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(train_config.device)
    src_pos = torch.from_numpy(src_pos).long().to(train_config.device)
    
    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha, p_alpha=p_alpha, e_alpha=e_alpha)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)

def get_data():
    tests = [ 
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
    ]
    data_list = list(text.text_to_sequence(test, train_config.text_cleaners) for test in tests)
    return data_list, tests

def get_wandb_table(model, WaveGlow):
    model.eval()
    data_list, texts = get_data()
    
    columns = ["Audio", "x Duration", "x Pitch", "x Energy", "Text"]
    data = []
    for speed in tqdm([0.8, 1., 1.3]):
        for pitch in [0.8, 1, 1.2]:
            for energy in [0.8, 1, 1.2]:
                for i, phn in enumerate(data_list):
                    mel, mel_cuda = synthesis(model, phn, speed, pitch, energy)

                    os.makedirs("tmp_results", exist_ok=True)

                    path = f"tmp_results/s={speed}_{pitch}_{energy}_{i}_waveglow.wav"
                    waveglow.inference.inference(
                        mel_cuda, WaveGlow,
                        f"tmp_results/s={speed}_{pitch}_{energy}_{i}_waveglow.wav"
                    )
                    data.append([wandb.Audio(path), speed, pitch, energy, texts[i]])
    table = wandb.Table(data=data, columns=columns)
    
    model.train()
    return table

def main(args):
    buffer = get_data_to_buffer(train_config)

    dataset = BufferDataset(buffer)

    training_loader = DataLoader(
        dataset,
        batch_size=train_config.batch_expand_size * train_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn_tensor,
        drop_last=True,
        num_workers=0
    )

    model = FastSpeech(model_config)
    model = model.to(train_config.device)

    fastspeech_loss = FastSpeechLoss()
    current_step = 0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9)

    scheduler = OneCycleLR(optimizer, **{
        "steps_per_epoch": len(training_loader) * train_config.batch_expand_size,
        "epochs": train_config.epochs,
        "anneal_strategy": "cos",
        "max_lr": train_config.learning_rate,
        "pct_start": 0.1
    })

    logger = WanDBWriter(train_config)

    WaveGlow = utils.get_WaveGlow()
    WaveGlow = WaveGlow.cuda()

    tqdm_bar = tqdm(total=train_config.epochs * len(training_loader) * train_config.batch_expand_size - current_step)
    model.train()
    try:
        for epoch in range(train_config.epochs):
            for i, batchs in enumerate(training_loader):
                # real batch start here
                for j, db in enumerate(batchs):
                    current_step += 1
                    tqdm_bar.update(1)
                    
                    logger.set_step(current_step)

                    # Get Data
                    character = db["text"].long().to(train_config.device)
                    mel_target = db["mel_target"].float().to(train_config.device)
                    duration = db["duration"].int().to(train_config.device)
                    mel_pos = db["mel_pos"].long().to(train_config.device)
                    src_pos = db["src_pos"].long().to(train_config.device)
                    pitch = db["pitch"].float().to(train_config.device)
                    energy = db["energy"].float().to(train_config.device)
                    max_mel_len = db["mel_max_len"]

                    # Forward
                    mel_output, duration_predictor_output, pitch_prediction, energy_prediction = model(character,
                                                                src_pos,
                                                                mel_pos=mel_pos,
                                                                mel_max_length=max_mel_len,
                                                                length_target=duration,
                                                                pitch_target=pitch,
                                                                energy_target=energy,
                                                                )

                    # Calc Loss
                    mel_loss, duration_loss, pitch_loss, energy_loss = fastspeech_loss(mel_output,
                                                            duration_predictor_output,
                                                            pitch_prediction,
                                                            energy_prediction,
                                                            mel_target,
                                                            duration,
                                                            pitch,
                                                            energy)
                    total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

                    # Logger
                    t_l = total_loss.detach().cpu().numpy()
                    m_l = mel_loss.detach().cpu().numpy()
                    d_l = duration_loss.detach().cpu().numpy()
                    p_l = pitch_loss.detach().cpu().numpy()
                    e_l = energy_loss.detach().cpu().numpy()
                    
                    logger.add_scalar("duration_loss", d_l)
                    logger.add_scalar("mel_loss", m_l)
                    logger.add_scalar("pitch_loss", p_l)
                    logger.add_scalar("energy_loss", e_l)
                    logger.add_scalar("total_loss", t_l)

                    # Backward
                    total_loss.backward()

                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(
                        model.parameters(), train_config.grad_clip_thresh)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    if current_step % train_config.save_step == 0:
                        os.makedirs(train_config.checkpoint_path, exist_ok=True)
                        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                        )}, os.path.join(train_config.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))

                        table = get_wandb_table(model, WaveGlow)
                        logger.wandb.log({"examples": table})
                        
                        print("save model at step %d ..." % current_step)
                        
                    if current_step > args.iters:
                        logger.wandb.finish()
    except KeyboardInterrupt:
        logger.wandb.finish()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="TTS Project")
    args.add_argument(
        "-i",
        "--iters",
        default=300_000,
        type=int,
        help="number of training iterations",
    )
    args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main(args)
