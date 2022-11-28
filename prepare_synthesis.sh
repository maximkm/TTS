#download checkpoints FastSpeech2
gdown "1kPtaj83geOTi6Cobp4kcKukQ335Exvhp&confirm=t"
gdown "1Sy5l-D4jP9M-2uUx2beDFKQbDW4pnmXc&confirm=t"

#download Waveglow
gdown "1tubyTrMyi4Djn-umdASnWcjq17y-72Ny&confirm=t"

mkdir checkpoints -p
mkdir src/waveglow/pretrained_model -p
mv finetune_model.pth checkpoints/
mv base_model.pth checkpoints/
mv waveglow_256channels.pt src/waveglow/pretrained_model/
