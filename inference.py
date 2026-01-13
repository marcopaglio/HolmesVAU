import argparse
import os
import torch
from decord import VideoReader, cpu
import matplotlib.pyplot as plt
from holmesvau.holmesvau_utils import load_model, generate, show_smapled_video


def parse_args():
    parser = argparse.ArgumentParser("HolmesVAU CLI Inference")

    parser.add_argument(
        "--video_path",
        type=str,
        default="./examples/robbery.mp4",
        help="Path to input video"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Could you specify the anomaly events present in the video?",
        help="Prompt text provided inline"
    )
    
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Path to text file containing the prompt"
    )

    parser.add_argument(
        "--use_ATS",
        action="store_true",
        help="Enable anomaly-focused temporal sampling (ATS)"
    )

    parser.add_argument(
        "--select_frames",
        type=int,
        default=12,
        help="Number of frames to sample"
    )

    return parser.parse_args()


def load_prompt(args):
    if args.prompt_file is not None:
        if not os.path.exists(args.prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
        with open(args.prompt_file, "r") as f:
            prompt = f.read().strip()
    else:
        prompt = args.prompt.strip()

    if len(prompt) == 0:
        raise ValueError("Prompt is empty")

    return prompt



def main():
    args = parse_args()

    # Load prompt
    prompt = load_prompt(args)
    
    video_path = args.video_path

    mllm_path = "./ckpts/HolmesVAU-2B"
    sampler_path = "./holmesvau/ATS/anomaly_scorer.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Load model
    model, tokenizer, generation_config, sampler = load_model(
        mllm_path,
        sampler_path,
        device
    )

    # Run inference
    pred, history, frame_indices, anomaly_score = generate(
        video_path,
        prompt,
        model,
        tokenizer,
        generation_config,
        sampler,
        select_frames=args.select_frames,
        use_ATS=args.use_ATS
    )

    print('\nUser:', prompt, '\nHolmesVAU:', pred)


    # Visualization
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    show_smapled_video(vr, frame_indices)

    # Save anomaly score plot
    if anomaly_score is not None:
        plt.figure(figsize=(8, 2))
        plt.plot(anomaly_score)
        for idx in frame_indices:
            plt.vlines(idx / 16, 0, 1, colors='r')
        plt.ylim(0, 1)
        plt.xlabel("Snippet")
        plt.ylabel("Anomaly Score")
        plt.title("Anomaly Score with Sampled Frames")
        plt.tight_layout()
        plt.savefig("anomaly_score_plot.png")
        plt.close()
        print("\nSaved anomaly_score_plot.png")


if __name__ == "__main__":
    main()
    