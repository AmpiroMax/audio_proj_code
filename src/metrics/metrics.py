import numpy as np
import librosa
from scipy.linalg import sqrtm
import torch
import torch.nn.functional as F
from transformers import (
    ClapProcessor,
    ClapModel,
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    AutoProcessor,
    HubertModel,
)


def load_audio(audio_info: dict, sr: int = 48000):
    audio, original_sr = librosa.load(audio_info["filepath"])
    start_sample = int(audio_info["start"] * original_sr)
    end_sample = int(audio_info["end"] * original_sr)

    trimmed_audio = audio[start_sample:end_sample]
    if original_sr != sr:
        trimmed_audio = librosa.resample(
            trimmed_audio, orig_sr=original_sr, target_sr=sr
        )

    return trimmed_audio, sr


def calculate_frechet_distance(out_emb, target_emb):
    mu_out, mu_target = np.mean(out_emb, axis=0), np.mean(target_emb, axis=0)

    if out_emb.shape[0] < 2:
        sigma_out = np.eye(out_emb.shape[1])
    else:
        sigma_out = np.cov(out_emb, rowvar=False)

    if target_emb.shape[0] < 2:
        sigma_target = np.eye(target_emb.shape[1])
    else:
        sigma_target = np.cov(target_emb, rowvar=False)

    diff = np.sum((mu_out - mu_target) ** 2)
    covmean, _ = sqrtm(sigma_out @ sigma_target, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = diff + np.trace(sigma_out + sigma_target - 2 * covmean)
    return fd


def calculate_kl_divergence(out_logits, target_logits):
    probs_out = F.softmax(out_logits, dim=-1)
    probs_target = F.softmax(target_logits, dim=-1)

    kl_div = F.kl_div(probs_out.log(), probs_target, reduction="batchmean")

    return kl_div.item()


def evaluate(
    out: dict,
    target: dict,
    clap_model,
    clap_processor,
    hubert_model,
    hubert_processor,
    sr=48000,
    device="cpu",
):
    """
    out and target have the following structure:
    {
        "filepath": "/path/to/music/file",
        "start": int(start-in-seconds),
        "end": int(end-in-seconds)
    }
    sr: sampling rate for loading audio (default: 32kHz).
    """
    clap_model = clap_model.to(device)

    hubert_model = hubert_model.to(device)

    out_audio, sr = load_audio(out, sr)
    out_inputs = clap_processor(
        audios=out_audio, sampling_rate=sr, return_tensors="pt", padding=True
    )
    out_inputs["input_features"] = out_inputs["input_features"].to(device)
    with torch.no_grad():
        out_emb = clap_model.get_audio_features(**out_inputs).cpu().numpy()

    target_audio, sr = load_audio(target, sr)
    target_inputs = clap_processor(
        audios=target_audio, sampling_rate=sr, return_tensors="pt", padding=True
    )
    target_inputs["input_features"] = target_inputs["input_features"].to(device)
    with torch.no_grad():
        target_emb = clap_model.get_audio_features(**target_inputs).cpu().numpy()

    fd = calculate_frechet_distance(out_emb, target_emb)

    out_audio, sr = load_audio(out, sr=16000)  # HuBert requires sr=16kHz
    out_inputs = hubert_processor(
        audio=out_audio, sampling_rate=sr, return_tensors="pt", padding=True
    )
    out_inputs["input_values"] = out_inputs["input_values"].to(device)
    out_inputs["attention_mask"] = out_inputs["attention_mask"].to(device)
    with torch.no_grad():
        out_logs = hubert_model(**out_inputs).last_hidden_state
    out_logs = out_logs.mean(dim=1)

    target_audio, sr = load_audio(target, sr=16000)
    target_inputs = hubert_processor(
        audio=target_audio, sampling_rate=sr, return_tensors="pt", padding=True
    )
    target_inputs["input_values"] = target_inputs["input_values"].to(device)
    target_inputs["attention_mask"] = target_inputs["attention_mask"].to(device)
    with torch.no_grad():
        target_logs = hubert_model(**target_inputs).last_hidden_state
    target_logs = target_logs.mean(dim=1)

    kl = calculate_kl_divergence(out_logs, target_logs)

    return {"FD": fd, "KL": kl}


def main():
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    clap_model.eval()

    hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
    hubert_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    hubert_model.eval()

    file1 = "/home/docker_miptml/image2music/audio_proj_code/dataset/music/beethoven_night_10_447.mp3"
    file2 = "/home/docker_miptml/image2music/audio_proj_code/dataset/music/debussi_sea_2_1_404.mp3"

    audio1 = {"filepath": file1, "start": 0, "end": 12}
    audio2 = {"filepath": file2, "start": 0, "end": 15}

    print(
        evaluate(
            audio1,
            audio2,
            clap_model,
            clap_processor,
            hubert_model,
            hubert_processor,
            device="cuda:0",
        )
    )


if __name__ == "__main__":
    main()
