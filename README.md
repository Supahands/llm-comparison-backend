                     
<h1 align="center" style="font-weight: bold;">LLM Comparison backend 💻</h1>

<p align="center">
<a href="#tech">Technologies</a>
<a href="#started">Getting Started</a>
<a href="#deployment">Deployment</a> •
<a href="#converter">GGUF Converter</a>
<a href="#routes">API Endpoints</a>
<a href="#colab">Collaborators</a>
<a href="#contribute">Contribute</a> 
</p>


<p align="center">This is an opensource project allowing you to compare two LLM's head to head with a given prompt, this section will be regarding the backend of this project, allowing for llm api's to be incorporated and used in the front-end</p>


<p align="center">
<a href="https://github.com/Supahands/llm-comparison-backend">📱 Visit this Project</a>
</p>
 
<h2 id="technologies">💻 Technologies</h2>

- Python 3.10+
- Modal serverless GPU's
- Poetry for dependency management
- llama.cpp
- HuggingFace Hub
 
<h2 id="started">🚀 Getting started</h2>

The majority of this project will be run via <a href="https://modal.com/">Modal</a> services, meaning all of the building and dependency installation will be handled by that
 
<h3>Prerequisites</h3>

Here you list all prerequisites necessary for running your project. For example:

- Python ^10.10
- [Modal pip package](https://modal.com/docs/guide)
- Poetry for dependency management
 
<h3>Cloning</h3>

How to clone your project

```bash
git clone https://github.com/Supahands/llm-comparison-backend.git
```
 
<h3>Starting</h3>

There are two components to this project, the `ollama` api server as well as the `litellm` server which will be what our frontend uses to connect to and retrieve different models from.

I have added both the applications into a single deploy file which can be run to allow both apps to be spun up at the same time using:

```bash
modal deploy --env dev deploy
```
 
<h3>Manual Deployment</h3>

**Production Deploy**:
```sh
modal deploy --env dev deploy
```

**Local Testing**:
```sh
modal serve --env dev deploy
```





<h2 id="converter">🔄 GGUF Converter</h2> <h3>Setup</h3>

1. Create Modal secret:
```sh 
modal secret create my-huggingface-secret HUGGING_FACE_HUB_TOKEN="your_token"
```

2. Run converter:
```sh
modal run --detach hugging_face_to_guff.py \
  --modelowner tencent \
  --modelname Tencent-Hunyuan-Large \
  --quanttype q8_0 \
  --username Supa-AI \
  --ollama-upload \
  --hf-upload
```
- The `--detach` command is used to allow this program to run even if your terminal disconnects from the modal servers
- `modelowner` is the repo owner that you are trying to get the model from
- `modelname` is the exact name of the model from that model owner you want to convert
- `quanttype` is the size of quantization, default is `q8_0` which is the largest this supports 
- `username` is used to determine which account it should upload to and create a repo for
- `ollama-upload` is a boolean determiner for whether it should also upload the newly created quantized models to ollama under your username.
    - **Important note!** Before uploading, make sure that the volume called `ollama` is created, once created you must run `ollama serve` on your own machine to retrieve the public and private sh keys to add to ollama, more details can be found [here](https://github.com/ollama/ollama/blob/main/docs/import.md)
- `hf-upload` another boolean determiner on whether it should upload these models to your hugging face repo 

<h3>Technical Details</h3>

#### Storage

- Uses Modal volumes (model-storage)
- Persists between runs and should use existing models when running again (will continue downloads from what it has as well)
- Supports large models (>10GB)

### Features

- Parallel downloads (8 connections) thanks to [booday's hugging face downloader](https://github.com/bodaay/HuggingFaceModelDownloader)
- Progress tracking with ETA
- Two-step conversion:
    1. FP16 format
    2. Quantization (Q4_K_M, Q5_K_M etc)

<h2 id="disclaimer">⚠️ Disclaimer</h2>

<p align="center">Currently, we do not support Anthropic models (Claude) on the official site due to API costs. We are actively seeking sponsors to help integrate these models. If you have suggestions for implementing Anthropic models or would like to contribute, please <a href="https://github.com/Supahands/llm-comparison-backend/issues/new">open an issue</a>!</p>

<p align="center">We welcome any creative solutions or partnerships that could help bring Anthropic model support to this comparison platform.</p>

### Conversion Process 
- Uses [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF conversion
- Two-step process:
    1. Convert to FP16 format
    2. Quantize to desired format (Q4_K_M, Q5_K_M etc)
- Supports importance matrix for optimized quantization
- Can split large models into manageable shards
 
<h2 id="colab">🤝 Collaborators</h2>

<p>Special thank you for all people that contributed for this project.</p>
<table>
<tr>

<td align="center">
<a href="https://github.com/OriginalByteMe">
<img src="https://avatars.githubusercontent.com/OriginalByteMe" width="100px;" alt="Noah Profile Picture"/><br>
<sub>
<b>Noah Rijkaard</b>
</sub>
</a>
</td>

</tr>
</table>
 
<h2 id="contribute">📫 Contribute</h2>

Here you will explain how other developers can contribute to your project. For example, explaining how can create their branches, which patterns to follow and how to open an pull request

1. `git clone https://github.com/Supahands/llm-comparison-backend`
2. `git checkout -b feature/NAME`
3. Follow commit patterns
4. Open a Pull Request explaining the problem solved or feature made, if exists, append screenshot of visual modifications and wait for the review!
 
<h3>Documentations that might help</h3>

[📝 How to create a Pull Request](https://www.atlassian.com/en/git/tutorials/making-a-pull-request)

[💾 Commit pattern](https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716)
