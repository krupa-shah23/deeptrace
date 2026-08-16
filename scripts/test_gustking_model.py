import os

# ------------------------------------------------------------
# Hugging Face / Transformers configuration
# ------------------------------------------------------------

# Prevent unnecessary parallelism from consuming extra memory.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import librosa

from transformers import (
    Wav2Vec2FeatureExtractor,
    AutoModelForAudioClassification,
)


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"

REAL_AUDIO = "data/demo/real_audio.wav"
GENERATED_AUDIO = "data/demo/generated_audio.wav"

TARGET_SAMPLE_RATE = 16000


# ============================================================
# Load pretrained model
# ============================================================

def load_model():

    print("=" * 70)
    print("LOADING PRETRAINED DEEPFAKE AUDIO MODEL")
    print("=" * 70)

    print()
    print("Model:")
    print(MODEL_ID)

    print()
    print("PyTorch version:", torch.__version__)
    print("Device: CPU")

    # --------------------------------------------------------
    # Feature extractor
    # --------------------------------------------------------

    print()
    print("Loading feature extractor...")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_ID
    )

    print("Feature extractor loaded successfully.")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Loading audio classification model...")
    print("This model is large. The first load may take some time.")
    print()

    try:

        model = AutoModelForAudioClassification.from_pretrained(
            MODEL_ID,

            # Explicitly keep the model on CPU.
            device_map=None,

            # Avoid automatic low-memory/meta-device loading.
            # This can be more reliable on Windows for this model.
            low_cpu_mem_usage=False,

            # Keep the original model precision.
            torch_dtype=torch.float32,
        )

    except OSError as error:

        print()
        print("=" * 70)
        print("MODEL LOADING FAILED")
        print("=" * 70)

        print()
        print("The model repository was reached successfully,")
        print("but Windows could not allocate enough virtual memory")
        print("to load the model weights.")

        print()
        print("Original error:")
        print(error)

        print()
        print("This is usually Windows error 1455:")
        print("The paging file is too small for this operation.")

        raise

    model.eval()

    print()
    print("=" * 70)
    print("MODEL LOADED SUCCESSFULLY")
    print("=" * 70)

    print()
    print("Number of labels:", model.config.num_labels)
    print("Label mapping:", model.config.id2label)

    print(
        "Feature extractor sampling rate:",
        processor.sampling_rate
    )

    print()

    return processor, model


# ============================================================
# Load audio
# ============================================================

def load_audio(audio_path):

    print("Loading audio:")
    print(audio_path)

    if not os.path.exists(audio_path):

        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    audio, sample_rate = librosa.load(
        audio_path,
        sr=TARGET_SAMPLE_RATE,
        mono=True,
    )

    duration = len(audio) / TARGET_SAMPLE_RATE

    print("Audio loaded successfully.")

    print(
        "Sample rate:",
        sample_rate
    )

    print(
        "Number of samples:",
        len(audio)
    )

    print(
        f"Duration: {duration:.3f} seconds"
    )

    print("Channels: 1 (mono)")

    print()

    return audio


# ============================================================
# Run inference
# ============================================================

def classify_audio(
    processor,
    model,
    audio_path,
):

    print("=" * 70)
    print("TESTING AUDIO")
    print("=" * 70)

    print()
    print("File:", audio_path)

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio = load_audio(audio_path)

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    print("Running feature extraction...")

    inputs = processor(
        audio,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )

    print("Feature extraction completed.")

    # --------------------------------------------------------
    # CPU inference
    # --------------------------------------------------------

    print()
    print("Running neural model inference...")

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    # --------------------------------------------------------
    # Convert logits to probabilities
    # --------------------------------------------------------

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )[0]

    predicted_class = torch.argmax(
        probabilities
    ).item()

    # --------------------------------------------------------
    # Label mapping
    # --------------------------------------------------------

    label_mapping = model.config.id2label

    predicted_label = label_mapping.get(
        predicted_class,
        str(predicted_class),
    )

    confidence = probabilities[
        predicted_class
    ].item()

    # --------------------------------------------------------
    # Display raw model output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL OUTPUT")
    print("=" * 70)

    print()

    print("Raw logits:")

    print(
        outputs.logits[0].tolist()
    )

    print()

    print("Class probabilities:")

    for class_id, probability in enumerate(
        probabilities
    ):

        class_name = label_mapping.get(
            class_id,
            str(class_id),
        )

        print(
            f"  Class {class_id} "
            f"({class_name}): "
            f"{probability.item():.6f}"
        )

    print()

    print("Predicted class:")
    print(predicted_class)

    print()

    print("Predicted label:")
    print(predicted_label)

    print()

    print("Confidence:")
    print(f"{confidence:.6f}")

    print()

    return {
        "audio_file": audio_path,
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": {
            label_mapping.get(
                i,
                str(i),
            ): probabilities[i].item()
            for i in range(
                len(probabilities)
            )
        },
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("GUSTKING DEEPFAKE AUDIO MODEL TEST")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Environment information
    # --------------------------------------------------------

    print("PyTorch version:")
    print(torch.__version__)

    print()

    print("CUDA available:")
    print(torch.cuda.is_available())

    print()

    print("Inference device:")
    print("CPU")

    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    processor, model = load_model()

    # --------------------------------------------------------
    # Test real audio
    # --------------------------------------------------------

    real_result = classify_audio(
        processor,
        model,
        REAL_AUDIO,
    )

    # --------------------------------------------------------
    # Test generated audio
    # --------------------------------------------------------

    generated_result = classify_audio(
        processor,
        model,
        GENERATED_AUDIO,
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print()

    print("REAL AUDIO")
    print("-" * 70)

    print(
        "File:",
        real_result["audio_file"]
    )

    print(
        "Prediction:",
        real_result["predicted_label"]
    )

    print(
        "Confidence:",
        f"{real_result['confidence']:.6f}"
    )

    print()

    print("GENERATED AUDIO")
    print("-" * 70)

    print(
        "File:",
        generated_result["audio_file"]
    )

    print(
        "Prediction:",
        generated_result["predicted_label"]
    )

    print(
        "Confidence:",
        f"{generated_result['confidence']:.6f}"
    )

    print()

    print("=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()