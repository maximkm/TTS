#download LjSpeech
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
mkdir data
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 src/data/LJSpeech-1.1

gdown "1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx&confirm=t"
mv train.txt src/data/

#download mels
gdown "1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j&confirm=t"
tar -xvf mel.tar.gz
mv mels src/data/

#download alignments
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip >> /dev/null
mv alignments src/data/

#download pitch and energy
gdown "1gjYCC2A7hmBsQtghHcymSLUg09qf4ab4&confirm=t"
gdown "1oRdvTuRKdRsYJt617Nloyq8qbbpHdbsc&confirm=t"
mv pitchs src/data/
mv energy src/data/
