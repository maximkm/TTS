from src.audio.stft import TacotronSTFT
from src.config import train_config
from src.dataloader import process_text
import pyworld as pw
from tqdm import tqdm
import numpy as np
import torch
import librosa
import time
import os

def get_energy_from_wav(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, energy = _stft.mel_spectrogram(audio)
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
    return energy

def process_pitch_energy(train_config):
    text = process_text(train_config.data_path)
    
    wavs_path = os.path.join('/'.join(train_config.data_path.split('/')[:-1]), "LJSpeech-1.1/wavs")
    wavs_names = sorted(os.listdir(wavs_path))
    
    hop_length = 256
    sampling_rate = 22050
    STFT = TacotronSTFT(1024, hop_length, 1024, 80, sampling_rate, 0, 8000)
    
    start = time.perf_counter()
    for i in tqdm(range(len(text))):
        mel_gt_name = os.path.join(train_config.mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(train_config.alignment_path, str(i)+".npy"))
        
        wav, sr = librosa.load(os.path.join(wavs_path, wavs_names[i]))
        pitch, t = pw.dio(
            wav.astype(np.float64),
            sr,
            frame_period=hop_length / sr * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sampling_rate)
        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None
        
        energy = get_energy_from_wav(wav.astype(np.float32), STFT)
        energy = energy[: sum(duration)]

        np.save(os.path.join(train_config.pitch_path, "ljspeech-pitch-%05d.npy" % (i+1)), pitch)
        np.save(os.path.join(train_config.energy_path, "ljspeech-energy-%05d.npy" % (i+1)), energy)

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))


if __name__ == '__main__':
    process_pitch_energy(train_config)
