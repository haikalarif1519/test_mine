from lstm_pipeline.inference.inference_pipeline import CONFIG, run_inference


if __name__ == "__main__":
    device = CONFIG["devices"][0]
    result = run_inference(device)
    if result is None:
        print(f"Inference failed for {device}")
    else:
        print(f"Inference result for {device}:")
        for step, probs in result.items():
            print(step, probs)
