from extract_formants import FeatureInput
import numpy as np

sources = [
    "C:/Users/lundb/Documents/Other/Music/i.wav",
    # "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/test_stuff/formant_test_0.7.wav", 
    "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/test_stuff/formant_test_1.wav",
    "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/test_stuff/formant_test_1.5.wav",
    ]

formant_feature =FeatureInput()

formant_list = []
for source in sources:
    formant_list.append(formant_feature.compute_formants(source))

print("formant mean table:")
for i, source in enumerate(sources):
    f = []
    for j in range(3):
        f.append(np.mean(formant_list[i][j]))
    print(source.split("/")[-1]+" - "+" ".join([str(s) for s in f]))

print("formant max table:")
for i, source in enumerate(sources):
    f = []
    for j in range(3):
        f.append(np.max(formant_list[i][j]))
    print(source.split("/")[-1]+" - "+" ".join([str(s) for s in f]))

print("formant min table:")
for i, source in enumerate(sources):
    f = []
    for j in range(3):
        f.append(np.min(formant_list[i][j][np.nonzero(formant_list[i][j])]))
    print(source.split("/")[-1]+" - "+" ".join([str(s) for s in f]))