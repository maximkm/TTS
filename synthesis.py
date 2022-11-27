import numpy as np
import torch

import argparse
import os
import sys
sys.path.append('src/')

from FastSpeech2 import FastSpeech
from config import model_config, train_config
import waveglow
import utils
import text


def synthesis(model, text, alpha=1.0, p_alpha=1.0, e_alpha=1.0):
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(train_config.device)
    src_pos = torch.from_numpy(src_pos).long().to(train_config.device)
    
    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha, p_alpha=p_alpha, e_alpha=e_alpha)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)

def main(args):
    model = FastSpeech(model_config)
    model = model.to(train_config.device)

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])

    WaveGlow = utils.get_WaveGlow()
    WaveGlow = WaveGlow.cuda()

    tests = []
    if args.text is not None:
        tests.append(args.text)
    if args.file is not None:
        with open(args.file) as f:
            for test in f.readlines():
                tests.append(test)
    
    assert len(tests) > 0, 'Need at least one text'
    phns = list(text.text_to_sequence(test, train_config.text_cleaners) for test in tests)

    model.eval()
    for i, phn in enumerate(phns):
        mel, mel_cuda = synthesis(model, phn, alpha=args.duration, p_alpha=args.pitch, e_alpha=args.energy)

        os.makedirs(args.output, exist_ok=True)

        name = f"{args.output}/waveglow_{i}.wav"
        waveglow.inference.inference(
            mel[None, ...].cuda(), WaveGlow,
            name
        )
        print(f'file "{name}" synthesis text: "{tests[i]}"')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="TTS Project")
    args.add_argument(
        "-c",
        "--checkpoint",
        default='checkpoints/finetune_model.pth',
        type=str,
        help="the path to the checkpoint of the model",
    )
    args.add_argument(
        "-d",
        "--duration",
        default=1.0,
        type=float,
        help="coefficient of sound length increase",
    )
    args.add_argument(
        "-p",
        "--pitch",
        default=1.0,
        type=float,
        help="coefficient increase in pitch",
    )
    args.add_argument(
        "-e",
        "--energy",
        default=1.0,
        type=float,
        help="coefficient increase in energy",
    )
    args.add_argument(
        "-f",
        "--file",
        default=None,
        type=str,
        help="the path to the file with texts for synthesis",
    )
    args.add_argument(
        "-t",
        "--text",
        default=None,
        type=str,
        help="text for synthesis",
    )
    args.add_argument(
        "-o",
        "--output",
        default="results",
        type=str,
        help="the path to save synthesized speech",
    )
    args = args.parse_args()
    main(args)
