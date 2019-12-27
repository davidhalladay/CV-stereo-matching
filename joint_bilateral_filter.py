import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r, border_type='reflect'):

        self.border_type = border_type
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s

    def kernel_r_transform(self, x):
        return np.exp(-x / 255. * x / 255. * self.factor_r)

    def joint_bilateral_filter(self, input, guidance):

        assert input.ndim == 3 , 'Dim of input is not valid.'
        assert guidance.ndim in [2,3] , 'Dim of guidance is not valid.'
        type = 'gray' if (guidance.ndim == 2) else 'rgb'

        # parameter Notebook
        output = np.zeros_like(input, dtype=float)
        r = int(np.ceil(3 * self.sigma_s))
        window_size = 2 * r + 1

        factor_s = 1. / (2 * self.sigma_s * self.sigma_s)
        factor_r = 1. / (2 * self.sigma_r * self.sigma_r)
        self.factor_r = factor_r

        # Image padding
        h, w, ch = input.shape
        I_pad = np.pad(input, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.float64)

        # Guidance padding
        if type == 'gray':
            h_G, w_G = guidance.shape
            G_pad = np.pad(guidance, ((r, r), (r, r)), 'symmetric').astype(np.float64)
        else:
            h_G, w_G, ch_G = guidance.shape
            G_pad = np.pad(guidance, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.float64)

        assert (h_G == h and w_G == w), 'Guidance and Input image mismatch.'

        # Generate spatial kernel
        x, y = np.meshgrid(np.arange(window_size) - r, np.arange(window_size) - r)
        kernel_s = np.exp(-(x * x + y * y) * factor_s)

        # Generate a table for range kernel
        # kernel_r_tab = np.exp(-np.arange(256) / 255. * np.arange(256) / 255. * factor_r)

        assert type in ['gray','rgb']
        if type == 'rgb':
            for y in range(0, h):
                for x in range(0, w):
                    # wgt = kernel_r_tab[abs(G_pad[y:y + window_size, x:x + window_size, 0] - G_pad[y+r, x+r, 0])] * \
                          # kernel_r_tab[abs(G_pad[y:y + window_size, x:x + window_size, 1] - G_pad[y+r, x+r, 1])] * \
                          # kernel_r_tab[abs(G_pad[y:y + window_size, x:x + window_size, 2] - G_pad[y+r, x+r, 2])] * \
                          # kernel_s
                    wgt = self.kernel_r_transform(G_pad[y:y + window_size, x:x + window_size, 0] - G_pad[y+r, x+r, 0]) * \
                          self.kernel_r_transform(G_pad[y:y + window_size, x:x + window_size, 1] - G_pad[y+r, x+r, 1]) * \
                          self.kernel_r_transform(G_pad[y:y + window_size, x:x + window_size, 2] - G_pad[y+r, x+r, 2]) * \
                          kernel_s
                    wacc = np.sum(wgt)

                    output[y, x, 0] = np.sum(wgt * I_pad[y:y + window_size, x:x + window_size, 0]) / wacc
                    output[y, x, 1] = np.sum(wgt * I_pad[y:y + window_size, x:x + window_size, 1]) / wacc
                    output[y, x, 2] = np.sum(wgt * I_pad[y:y + window_size, x:x + window_size, 2]) / wacc

        if type == 'gray':
            for y in range(0, h):
                for x in range(0, w):
                    # wgt = kernel_r_tab[abs(G_pad[y:y + window_size, x:x + window_size] - G_pad[y+r, x+r])] * kernel_s
                    wgt = self.kernel_r_transform(G_pad[y:y + window_size, x:x + window_size] - G_pad[y+r, x+r]) * kernel_s
                    wacc = np.sum(wgt)
                    output[y, x, 0] = np.sum(wgt * I_pad[y:y + window_size, x:x + window_size, 0]) / wacc
                    output[y, x, 1] = np.sum(wgt * I_pad[y:y + window_size, x:x + window_size, 1]) / wacc
                    output[y, x, 2] = np.sum(wgt * I_pad[y:y + window_size, x:x + window_size, 2]) / wacc

        return output
