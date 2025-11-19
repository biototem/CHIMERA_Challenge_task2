import albumentations as A
import random
import numpy as np
class GradualBrightness(A.ImageOnlyTransform):
    '''
    梯度渐变亮度变化
    '''

    def __init__(self, always_apply=False, p=0.5):
        super(GradualBrightness, self).__init__(always_apply=always_apply, p=p)

    def get_transform_init_args_names(self):
        return ()

    def apply(self, img, **params):
        x = img
        v = random.choice(['+x', '-x', '+y', '-x'])
        grad_range = [random.uniform(0.8, 0.99), 1.]

        if v == '+x':
            g = np.linspace(grad_range[0], grad_range[1], x.shape[1])
            g = g[None, :, None]
        elif v == '-x':
            g = np.linspace(grad_range[1], grad_range[0], x.shape[1])
            g = g[None, :, None]
        elif v == '+y':
            g = np.linspace(grad_range[0], grad_range[1], x.shape[0])
            g = g[:, None, None]
        elif v == '-y':
            g = np.linspace(grad_range[1], grad_range[0], x.shape[0])
            g = g[:, None, None]
        else:
            raise AssertionError('Error! Bad vec param in GradualBrightness.')

        y = x * g
        y = y.astype(x.dtype)
        return y


class MyAugFunc:
    def __init__(self):
        self.aug = A.Compose([
            A.ImageCompression(85, 100),
            A.RandomRotate90(),
            A.RandomGamma(),
            A.ColorJitter(0.1, 0.05, 0.02, 0.02),
            GradualBrightness(),
        ])

    def __call__(self, x):
        y = self.aug(image=x)['image']
        return y
