import numpy as np

_ShadowEnd = 160
_HighlightStart = 200


def _gammaCurve(gamma, x):
    """ Returns from 0.0 to 1.0"""
    return pow((x / 255.0), (1.0 / gamma))


def _calcDevelopment(shadow_level, high_level, x):
    """
This function returns a development like this:

 (return)
    ^  
    |
0.5 |                 o   -   o                  <-- mids level, always 0.5
    |             -               -     
    |          -                       -      
    |       -                              o     <-- high_level  eg. 0.25
    |    -                                       
    | o                                          <-- shadow_level eg. 0.15
    |   
 0 -+-----------------|-------|------------|----->  x  (input)
    0                160     200          255
    """
    if x < _ShadowEnd:
        power = 0.5 - (_ShadowEnd - x) * (0.5 - shadow_level) / _ShadowEnd
    elif x < _HighlightStart:
        power = 0.5
    else:
        power = 0.5 - (x - _HighlightStart) * (0.5 - high_level) / (255 - _HighlightStart)

    return power

class Map:
    def __init__(self, map):
        self.map = map

    @staticmethod
    def calculate(src_gamma, noise_power, shadow_level, high_level) -> 'Map':
        map = np.zeros([256, 256], dtype=np.uint8)

        # We need to level off top end and low end to leave room for the noise to breathe
        crop_top = noise_power * high_level / 12
        crop_low = noise_power * shadow_level / 20

        pic_scale = 1 - (crop_top + crop_low)
        pic_offs = 255 * crop_low

        for src_value in range(0, 256):
            # Gamma compensate picture source value itself
            pic_value = _gammaCurve(src_gamma, src_value) * 255.0

            # In the shadows we want noise gamma to be 0.5, in the highs, 2.0:
            gamma = pic_value * (1.5 / 256) + 0.5
            gamma_offset = _gammaCurve(gamma, 128)
            
            # Power is determined by the development
            power = _calcDevelopment(shadow_level, high_level, pic_value)

            for noise_value in range(0, 256):
                gamma_compensated = _gammaCurve(gamma, noise_value) - gamma_offset
                value = pic_value * pic_scale + pic_offs + 255.0 * power * noise_power * gamma_compensated
                if value < 0:
                    value = 0
                elif value < 255.0:
                    value = int(value)
                else:
                    value = 255
                map[src_value, noise_value] = value

        return Map(map)

    def lookup(self, pic_value, noise_value):
        return self.map[pic_value, noise_value]

    def saveToFile(self, filename):
        from PIL import Image
        img = Image.fromarray(self.map)
        img.save(filename)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    def plotfunc(x_min, x_max, step, func):
        x_all = np.arange(x_min, x_max, step)
        y = []
        for x in x_all:
            y.append(func(x))

        plt.figure()
        plt.plot(x_all, y)
        plt.grid()
        
    def development1(x):
        return _calcDevelopment(0.2, 0.3, x)

    def gamma05(x):
        return _gammaCurve(0.5, x)
    def gamma1(x):
        return _gammaCurve(1, x)
    def gamma2(x):
        return _gammaCurve(2, x)

    plotfunc(0.0, 255.0, 1.0, development1)
    plotfunc(0.0, 255.0, 1.0, gamma05)
    plotfunc(0.0, 255.0, 1.0, gamma1)
    plotfunc(0.0, 255.0, 1.0, gamma2)
    plt.show()