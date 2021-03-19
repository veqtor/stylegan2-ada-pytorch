import dnnlib
import math
import moviepy.editor
import time
from numpy import linalg
import torch
import numpy as np
import pickle
import legacy

import glob
import os


def main():
    trunc_psi = 0.9


    files = glob.glob("training-runs/*/*.pkl")
    files.sort(key=os.path.getmtime)

    device = torch.device('cuda')
    with dnnlib.util.open_url(files[-1]) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    cancelled = False
    while not cancelled:
        try:
            rnd = np.random
            latents_a = rnd.randn(1, G.z_dim)
            latents_b = rnd.randn(1, G.z_dim)
            latents_c = rnd.randn(1, G.z_dim)

            def circ_generator(latents_interpolate):
                radius = 40.0

                latents_axis_x = (latents_a - latents_b).flatten() / linalg.norm(latents_a - latents_b)
                latents_axis_y = (latents_a - latents_c).flatten() / linalg.norm(latents_a - latents_c)

                latents_x = math.sin(math.pi * 2.0 * latents_interpolate) * radius
                latents_y = math.cos(math.pi * 2.0 * latents_interpolate) * radius

                latents = latents_a + latents_x * latents_axis_x + latents_y * latents_axis_y
                return latents

            def mse(x, y):
                return (np.square(x - y)).mean()

            def convert_img(img):
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                if img.shape[-1] == 1:
                    return np.repeat(img, 3, -1)
                return img

            def generate_from_generator_adaptive(gen_func):
                max_step = 0.5
                current_pos = 0.0
                max_tries = 200

                change_min = 4.0
                change_max = 4.4

                current_latent = torch.from_numpy(gen_func(current_pos)).to(device)

                current_image = G(current_latent, torch.zeros([1, G.c_dim], device=device), truncation_psi=trunc_psi, noise_mode='const')

                current_image = convert_img(current_image)
                array_list = []

                video_length = 1.0
                while(current_pos < video_length):
                    array_list.append(current_image)

                    lower = current_pos
                    upper = current_pos + max_step
                    current_pos = (upper + lower) / 2.0

                    current_latent = torch.from_numpy(gen_func(current_pos)).to(device)
                    current_image = G(current_latent, torch.zeros([1, G.c_dim], device=device), truncation_psi=trunc_psi, noise_mode='const')
                    current_image = convert_img(current_image)
                    current_mse = mse(array_list[-1], current_image)
                    i = 0
                    while (current_mse < change_min or current_mse > change_max) and i < max_tries:
                        if current_mse < change_min:
                            lower = current_pos
                            current_pos = (upper + lower) / 2.0

                        if current_mse > change_max:
                            upper = current_pos
                            current_pos = (upper + lower) / 2.0


                        current_latent = torch.from_numpy(gen_func(current_pos)).to(device)
                        current_image = G(current_latent, torch.zeros([1, G.c_dim], device=device), truncation_psi=trunc_psi, noise_mode='const')
                        current_image = convert_img(current_image)
                        current_mse = mse(array_list[-1], current_image)
                        i += 1
                    print(current_pos, current_mse)
                if current_pos >= 1.:
                    return array_list[:-1]
                return array_list

            frames = generate_from_generator_adaptive(circ_generator)
            frames = moviepy.editor.ImageSequenceClip(frames, fps=60)

            # Generate video.
            mp4_file = 'results/%d.mp4' % int(time.time())
            mp4_codec = 'libx264'
            mp4_bitrate = '10M'
            mp4_fps = 60

            frames.write_videofile(mp4_file, fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)
        except KeyboardInterrupt:
            cancelled = True
            raise

if __name__ == "__main__":
    main()