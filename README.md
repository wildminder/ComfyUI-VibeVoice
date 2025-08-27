<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<div align="center">
  <h1 align="center">ComfyUI-VibeVoice</h1>

<img src="./example_workflows/VibeVoice_example.png" alt="ComfyUI-VibeVoice Nodes" alt="Logo" width="600" height="388">

  <p align="center">
    A custom node for ComfyUI that integrates Microsoft's VibeVoice, a frontier model for generating expressive, long-form, multi-speaker conversational audio.
    <br />
    <br />
    <a href="https://github.com/wildminder/ComfyUI-VibeVoice/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/wildminder/ComfyUI-VibeVoice/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>

<!-- PROJECT SHIELDS -->
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

This project brings the power of **VibeVoice** into the modular workflow of ComfyUI. VibeVoice is a novel framework by Microsoft for generating expressive, long-form, multi-speaker conversational audio. It excels at creating natural-sounding dialogue, podcasts, and more, with consistent voices for up to 4 speakers.

The custom node handles everything from model downloading and memory management to audio processing, allowing you to generate high-quality speech directly from a text script and reference audio files.

**Key Features:**
*   **Multi-Speaker TTS:** Generate conversations with up to 4 distinct voices in a single audio output.
*   **Zero-Shot Voice Cloning:** Use any audio file (`.wav`, `.mp3`) as a reference for a speaker's voice.
*   **Automatic Model Management:** Models are downloaded automatically from Hugging Face and managed efficiently by ComfyUI to save VRAM.
*   **Fine-Grained Control:** Adjust parameters like CFG scale, temperature, and sampling methods to tune the performance and style of the generated speech.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

Follow these steps to get the ComfyUI-VibeVoice node running in your environment.

### Installation
The node can be installed via **Use ComfyUI Manager:** Find the ComfyUI-VibeVoice and install it. Or install it manually
    
1.  **Clone the Repository:**
    Navigate to your `ComfyUI/custom_nodes/` directory and clone this repository:
    ```sh
    git clone https://github.com/your-username/ComfyUI-VibeVoice.git
    ```

2.  **Install Dependencies:**
    Open a terminal or command prompt, navigate into the cloned directory, and install the required Python packages:
    ```sh
    cd ComfyUI-VibeVoice
    pip install -r requirements.txt
    ```

3.  **Start/Restart ComfyUI:**
    Launch ComfyUI. The "VibeVoice TTS" node will appear under the `audio/tts` category. The first time you use the node, it will automatically download the selected model to your `ComfyUI/models/tts/VibeVoice/` folder.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

The node is designed to be intuitive within the ComfyUI workflow.

1.  **Add Nodes:** Add the `VibeVoice TTS` node to your graph. Use ComfyUI's built-in `Load Audio` node to load your reference voice files.
2.  **Connect Voices:** Connect the `AUDIO` output from each `Load Audio` node to the corresponding `speaker_*_voice` input on the VibeVoice TTS node.
3.  **Write Script:** In the `text` input, write your dialogue. Assign lines to speakers using the format `Speaker 1: ...`, `Speaker 2: ...`, etc., on separate lines.
4.  **Generate:** Queue the prompt. The node will process the script and generate a single audio file containing the full conversation.

_For a complete workflow, you can drag the example image `images/vibevoice_workflow_example.png` onto your ComfyUI canvas._

### Node Inputs

*   **`model_name`**: Select the VibeVoice model to use (e.g., `VibeVoice-1.5B`). It will be downloaded automatically if not found.
*   **`text`**: The conversational script. Lines must be prefixed with `Speaker <number>:` (e.g., `Speaker 1:`). The numbers correspond to the speaker voice inputs.
*   **`cfg_scale`**: Controls how strongly the model adheres to the reference voice's timbre. Higher values improve voice similarity but may reduce naturalness.
*   **`inference_steps`**: Number of diffusion steps. More steps can improve quality but increase generation time.
*   **`seed`**: A seed for reproducibility. Set to 0 for a random seed on each run.
*   **`do_sample`**: Toggles between deterministic (greedy) and stochastic (sampling) generation. Enable for more varied and potentially more natural-sounding speech.
*   **`temperature`**: Controls randomness when sampling. Higher values lead to more diverse outputs. (Only active if `do_sample` is enabled).
*   **`top_p` / `top_k`**: Nucleus and Top-K sampling parameters to further control the randomness of the generation. (Only active if `do_sample` is enabled).
*   **`speaker_*_voice` (Optional)**: Connect an `AUDIO` output from a `Load Audio` node to provide a voice reference for the corresponding speaker number in the script.

### Tips from the Original Authors

*   **Punctuation:** For Chinese text, using English punctuation (commas and periods) can improve stability.
*   **Model Choice:** The 7B model variant (`VibeVoice-Large-pt`) is generally more stable.
*   **Spontaneous Sounds/Music:** The model may spontaneously generate background music, especially if the reference audio contains it or if the text includes introductory phrases like "Welcome to...". This is an emergent capability and cannot be directly controlled.
*   **Singing:** The model was not trained on singing data, but it may attempt to sing as an emergent behavior. Results may vary.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

This project is distributed under the MIT License. See `LICENSE.txt` for more information. The VibeVoice model and its components are subject to the licenses provided by Microsoft. Please use responsibly.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

*   **Microsoft** for creating and open-sourcing the [VibeVoice](https://github.com/microsoft/VibeVoice) project.
*   **The ComfyUI team** for their incredible and extensible platform.
*   **othneildrew** for the [Best-README-Template](https://github.com/othneildrew/Best-README-Template).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/wildminder/ComfyUI-VibeVoice.svg?style=for-the-badge
[contributors-url]: https://github.com/wildminder/ComfyUI-VibeVoice/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/wildminder/ComfyUI-VibeVoice.svg?style=for-the-badge
[forks-url]: https://github.com/wildminder/ComfyUI-VibeVoice/network/members
[stars-shield]: https://img.shields.io/github/stars/wildminder/ComfyUI-VibeVoice.svg?style=for-the-badge
[stars-url]: https://github.com/wildminder/ComfyUI-VibeVoice/stargazers
[issues-shield]: https://img.shields.io/github/issues/wildminder/ComfyUI-VibeVoice.svg?style=for-the-badge
[issues-url]: https://github.com/wildminder/ComfyUI-VibeVoice/issues
