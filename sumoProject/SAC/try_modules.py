from sumoProject.SAC.model import ImageEncoder, ImageDecoder, LSTMPredictor
import torch

kernel_size = (7, 3)
padding = (3, 1)
input_image_size = (3, 180, 18)
max_pool = (2, 2)
hidden_channels = 128
hidden_lstm = 256
batch_size = 64
sample_image = torch.ones(input_image_size)

sample_batch_image = torch.stack([sample_image for _ in range(batch_size)])
encoder = ImageEncoder(in_features=input_image_size, out_features=(1, 20, 2), hidden_size=hidden_channels,
                       kernel_size=kernel_size, padding=padding, stride=max_pool)
encoder_output = encoder(sample_batch_image)
decoder = ImageDecoder(in_features=encoder_output.size()[-1], out_features=input_image_size,
                       hidden_channels=hidden_channels,
                       stride=max_pool, kernel_size=kernel_size, padding=padding)
output = decoder(encoder_output)
predict = LSTMPredictor(latent_space=encoder_output.flatten(1).size()[-1], hidden_size=hidden_lstm)
prediction = predict(encoder_output)
print()



