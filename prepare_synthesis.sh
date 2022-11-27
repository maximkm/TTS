#download checkpoints FastSpeech2
gdown https://drive.google.com/file/d/1kPtaj83geOTi6Cobp4kcKukQ335Exvhp
gdown https://drive.google.com/file/d/1Sy5l-D4jP9M-2uUx2beDFKQbDW4pnmXc

#download Waveglow
gdown https://drive.google.com/file/d/1tubyTrMyi4Djn-umdASnWcjq17y-72Ny

mv finetune_model.pth checkpoints/
mv base_model.pth checkpoints/
mv waveglow_256channels.pt src/waveglow/pretrained_model/
