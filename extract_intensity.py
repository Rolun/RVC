import os
import parselmouth
import numpy as np

base_directory = "C:/Users/lundb/Documents/Other/Music/Martin_recordings"
files = os.listdir(base_directory)

intensities = []

for file in files:
    path = os.path.join(base_directory, file)
    snd = parselmouth.Sound(path)
    intensity = snd.to_intensity()
    intensities.append(np.asarray(intensity.values))

import pdb; pdb.set_trace()