import os
from TTS.bin.compute_embeddings import compute_embeddings

def rvc_formatter(root_path, manifest_file, **kwargs):
    """Assumes each line as ```<filename>|...|speaker_id```
    """
    txt_file = os.path.join(root_path, manifest_file)
    items = []
    with open(root_path, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = cols[0]
            if "mute" in wav_file:
                continue
            text = ""
            speaker_name = cols[-1]
            items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name, "root_path": root_path})
    
    return items

compute_embeddings(
    model_path = "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/speaker_embeddings/model_se.pth", 
    config_path = "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/speaker_embeddings/config_se.json", 
    output_path = "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/speaker_embeddings/speakers.json", 
    formatter = rvc_formatter,
    dataset_name = "mixed_dataset",
    dataset_path = "C:/Users/lundb/Documents/Other/Music/RVC-beta/RVC-beta-v2-0528/logs/merged2/filelist.txt"
    )