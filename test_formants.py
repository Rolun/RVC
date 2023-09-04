from extract_formants import FeatureInput as FormantFeatureInput
from extract_f0_print import FeatureInput as PitchFeatureInput
import numpy as np
import matplotlib.pyplot as plt

# sources = [
#     "C:/Users/lundb/Documents/Other/Music/i.wav", # GT
#     "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/test_stuff/formant_test_encoder_all_i_0.7.wav", 
#     "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/test_stuff/formant_test_encoder_all_i_1.wav",
#     "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/test_stuff/formant_test_encoder_all_i_1.5.wav",
#     ]

sources = [
    "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/test_stuff/formant_test_short_baseline.wav", # GT
    # "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/test_stuff/formant_test_encoder_all_0.7_short.wav", 
    "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/test_stuff/formant_test_encoder_all_1_short.wav",
    "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/test_stuff/formant_test_encoder_all_f-5_1.wav",
    # "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/test_stuff/formant_test_encoder_all_1.5_short.wav",
    ]

formant_feature = FormantFeatureInput()
pitch_feature = PitchFeatureInput()

formant_list = []
pitch_list = []
for source in sources:
    formant_list.append(formant_feature.compute_formants(source))
    pitch_list.append(pitch_feature.compute_f0(source, "crepe", 160))

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

print("First 20 formant values:")
for i, source in enumerate(sources):
    print("From: ", source)
    f = []
    for j in range(3):
        print(formant_list[i][j][np.nonzero(formant_list[i][j])][:20])
        

plt.figure(figsize=(12, 6))

for i, source in enumerate(sources):
    plt.subplot(len(sources), 1, 1+i)
    plt.plot(np.arange(len(pitch_list[i])), pitch_list[i], label="f0", linestyle='-')
    plt.plot(np.arange(len(formant_list[i][0])), formant_list[i][0], label="f1", linestyle='-')
    plt.plot(np.arange(len(formant_list[i][1])), formant_list[i][1], label="f2", linestyle='-')
    plt.plot(np.arange(len(formant_list[i][2])), formant_list[i][2], label="f3", linestyle='-')
    plt.title(source)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.grid(True)

plt.show()