#download LjSpeech
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
mkdir data
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 data/LJSpeech-1.1

gdown https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx
mv train.txt data/

#download mels
gdown https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j
tar -xvf mel.tar.gz
mv mels src/data/

#download alignments
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip >> /dev/null
mv alignments src/data/

#download pitch and energy
gdown https://drive.google.com/file/d/1gjYCC2A7hmBsQtghHcymSLUg09qf4ab4
gdown https://drive.google.com/file/d/1oRdvTuRKdRsYJt617Nloyq8qbbpHdbsc
mv pitchs src/data/
mv energy src/data/
