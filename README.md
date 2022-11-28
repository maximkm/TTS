# TTS project
The project is made for educational purposes, as the homework of the course [deep learning for audio processing](https://github.com/markovka17/dla).

## Installation guide
It is recommended to use python 3.8 or 3.9

You need to clone the repository and install the libraries:
```shell
git clone https://github.com/maximkm/TTS.git
cd TTS
pip install -r requirements.txt
```

## Speech synthesis

1/2) To synthesize speech, you need to download 2 checkpoints for FastSpeech2 and a pre-trained WaveGlow model.
This can be done with an automated script:

```bash
bash prepare_synthesis.sh
```

2/2) For the synthesis, it is enough to run the script `synthesis.py` to view all the startup arguments, you need to run it with the argument `--help`

A few examples of running a script:

```bash
python synthesis.py -f TTS.txt
```

This script loads the FastSpeech2 checkpoint after training for 273k iterations, synthesizes all the texts that are in the file `TTS.txt` and will save them to the `results` directory.

```bash
python synthesis.py -c checkpoints/base_model.pth -f TTS.txt -t "one two free" -d 0.8 -p 1.3 -e 1.3 -o test_synthesis
```

And this script will run synthesis with a checkpoint from the report, after learning 160k iterations, in addition to `TTS.txt` synthesizes the text "one two free", `speeds` up speech by 20%, and also increases `energy` and `pitch` by 30% and saves all files to the `test_synthesis` folder.

## Reproducing learning

To train the model, you will need to download the LJSpeech dataset, as well as pre-generated mels, alignments, pitch and energy. All this can be done with an automated script:

```bash
bash prepare_train.sh
```

You can also generate Pitch and Energy yourself, just run the script below

```bash
python prepare_pitch-energy.py
```

Finally, to start the training, it is enough to run the script:

```bash
python train.py
```

More details about training and experiments are written in the report: [Wandb report](https://wandb.ai/maximkm/fastspeech/reports/FastSpeech2-report--VmlldzozMDM5NTA5?accessToken=szlvp1en2h0oxymcx207qz5z8yuv90nn3va5zr6qh8twzn6qggehe6hf2nxu7l17)

## Credits

This repository contains a sub repository [FastSpeech](https://github.com/xcmyz/FastSpeech).
