import sox
from pathlib import Path


class WavPreprocessor:
    def __init__(
        self,
        samplerate=16000,
    ):
        self.tfm = sox.Transformer()
        self.tfm.convert(samplerate=samplerate)

    def downsample(
        self,
        input_filepath,
        output_filepath,
    ):
        if not input_filepath.exists():
            print(f"Error: Not found {input_filepath}")
            return

        self.tfm.build_file(
            input_filepath=input_filepath,
            output_filepath=output_filepath,
        )


if __name__ == "__main__":
    wav_preprocessor = WavPreprocessor()

    original_wav_dir = Path("../data/original/jsut_ver1.1/basic5000/wav")
    out_wav_dir = Path("../data/wav")
    out_scp_dir = Path("../data/label/all")
    out_wav_dir.mkdir(exist_ok=True)
    out_scp_dir.mkdir(exist_ok=True)

    with open(original_wav_dir / "wav.scp", mode="w") as scp_file:
        for i in range(5000):
            filename = f"BASIC5000_{i+1:04d}"
            wav_path_in = original_wav_dir / (filename + ".wav")
            wav_path_out = out_wav_dir / (filename + ".wav")
            print(wav_path_in)
            wav_preprocessor.downsample(
                input_filepath=wav_path_in,
                output_filepath=wav_path_out,
            )

            scp_file.write(f"{filename} {wav_path_out.absolute()}\n")
