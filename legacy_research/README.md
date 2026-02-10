# Tidal Language Model

The **Tidal Language Model** learns through physics. During training, concepts from the vocabulary are N-bodies in a **512D** *embedding space*. This is projected to an **8D** *Semantic Space* which is then projected to a **2D** *physical space* where a simulation is applied to the N-bodies. The new positions are then projected back to the **8D** *Semantic Space* where they are affected by an endocrine and “tide” or tick regulated resonance system. Physical loss is calculated at this layer using a custom *Grand Unifying Loss* function that is based on Helmholtz Free Energy. Finally, these new positions are projected back to the final **512D** output. In this way, we aim to create a *Semantic Space*, influenced by hormonal responses to periodic changes in “tide” and sequential and cognitive-like triggers from the input.

## Semantic Space

Theoretically, the *Semantic Space* mimics the following planes and their axes:

1. **Core Conceptual Plane** defines the intrinsic nature of a concept.  
   1. **G-Axis (Groundedness)** Measures the concept's tie to direct sensory-motor experience.  
      1. **Low G** Concrete & Sensory (*rock, water, heavy*)  
      2. **High G** Abstract & Relational (*justice, theory, freedom*)    
   2. **X-Axis (Taxonomic Specificity)** Measures the concept's level in a hierarchy.  
      1. **Low X** General & Superordinate (*animal, tool, idea*)  
      2. **High X** Specific & Subordinate (*poodle, screwdriver, sonnet*)  
2. **Affective Plane (Emotion)** maps the core emotional qualities based on the Circumplex Model.  
   1. **V-Axis (Valence)** Measures the positive or negative emotional charge.  
      1. **Low V** Negative (*pain, sad, anger*)  
      2. **High V** Positive (*joy, peace, beauty*)   
   2. **A-Axis (Arousal)** Measures the emotional intensity or energy.  
      1. **Low A** Deactivated & Calm (*sleep, bored, serene*)  
      2. **High A** Activated & Excited (*panic, rage, surprise*)  
3. **Interoceptive Plane (Bodily State)** maps the underlying physiological states, inspired by the gut-brain axis.  
   1.  **H-Axis (Homeostasis)** Measures the sense of internal balance versus dysregulation.  
      1. **Low H** Dysregulated (*nausea, fatigue, stress*)  
      2. **High H** Regulated (*healthy, rested, energized*)  
   2. **S-Axis (Somatic Focus)** Measures if a concept is experienced more in the body or mind.  
      1. **Low S** Cognitive (*belief, memory, curiosity*)  
      2. **High S** Somatic/Physical (*pain, itch, warmth*)  
4. **Structural Plane (Contextual Role)** defines the concept's role within a larger structure like a sentence or timeline.  
   1. **F-Axis (Functional Role)** Measures a word's purpose as either providing meaning or grammatical structure.  
      1. **Low F** Lexical/Content (*mountain, sun, ocean*)  
      2. **High F** Functional/Grammar (*is, the, because*)  
   2. **T-Axis (Temporal Orientation)** Measures a concept's inherent relationship to time.  
      1. **Low T** Past-Oriented (*history, memory, yesterday*)  
      2. **High T** Future-Oriented (plan, hope, tomorrow)

See [`TidalLanguageModel.py`](TidalLanguageModel.py) for more details.

## *Grand Unifying Loss* Function

The final loss is a contrastive one, where we want the free energy of a correct context `(Fpos​)` to be lower than that of a random context `(Fneg​)`

```math
L=mean(relu(F_{pos}−F_{neg}+M))
```

Where the Free Energy, `F`, is defined as:

```math
F=\underbrace{(PE_{pairwise}+PE_{well})}_{Internal Energy (U)}−\underbrace{T⋅S}_{Entropy Term}
```

Where `PE_pairwise` is the pairwise potential energy and `PE_well` is the well potential energy. `T` is the temperature and `S` is the entropy.

See [`Physics2D.py`](Physics2D.py) for more details.

## Building the Vocabulary

The vocabulary is built from the foundational corpus file using **lemmatization**, with the `spaCy` library. **Lemmatization** is the process of converting a word to its base form. For example, the word "running" would be converted to "run". This creates the "concepts" that the model will project into the *Semantic Space*. 

See [`Preprocess.py`](Preprocess.py) for more details.

## Semantic Endocrine System

The *Semantic Endocrine System* releases hormones in response to sequence and cognitive-like triggers from the input. 

See `detect_triggers`, `release_hormones`, and `apply_hormonal_effects` methods in [`SemanticEndocrineSystem.py`](SemanticEndocrineSystem.py) for more details.

## Development

### Project Structure

```plaintext
tidal-language-model/
├── cache/
├── configs/
   ├── base_config.yaml
├── corpus/
   ├── foundational.txt
   ├── high_tide.txt
   ├── low_tide.txt
   ├── storm_tide.txt
├── experiments/
├── logs/
   ├── training-endocrine-system-*.log
   ├── evaluation-endocrine-system-*.log
   ├── training-physics-*.log
   ├── evaluation-physics-*.log
├── tests/
   ├── test_physics.py
   ├── test_semantic_endocrine_system.py
   ├── test_tidal_language_model.py
├── tidal-env/
├── .gitignore
├── AssociativeDataset.py
├── DynamicLRScheduler.py
├── Evaluator.py
├── Generator.py
├── Main.py
├── Physics2D.py
├── Preprocess.py
├── requirements.txt
├── README.md
├── SemanticEndocrineSystem.py
├── TidalLanguageModel.py
├── Trainer.py
├── Utils.py
```

### Setup Environment

On Windows, WSL2 is recommended with any Linux distro, the default Ubuntu installation is fine. Use a WSL2 Terminal instance and run the following commands from the root directory of the project:

```bash
python3 -m venv tidal-env && \
source tidal-env/bin/activate && \
sudo apt-get update && \
sudo apt-get install build-essential python3.12-venv python3.12-dev && \
pip3 install -r requirements.txt
```

Then install these remaining dependencies:

#### CUDA

The model requires CUDA support.

Set the CUDA version according to your GPU.

```bash
CUDA_VERSION="cu129"
```

Then run the following command:

```bash
pip3 install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
```

#### Tokenizer

The spaCy model is used to tokenize the input text as well as lemmatize it. 

Download the spaCy model:

```bash
python3 -m spacy download en_core_web_sm
```

### Train the Model

First, preprocess the vocabulary and cache the training pairs:

```bash
python3 Preprocess.py
```

Then, set the configuration for the training run:

```bash
CONFIG_FILE="configs/base_config.yaml"
```

Finally, train the model:

```bash
python3 Main.py --config "${CONFIG_FILE}"
```

To start the environment, run the following command:

```bash
source tidal-env/bin/activate
```

To deactivate the environment, run the following command:

```bash
deactivate
```

### Generate Text

Set the following variables, replacing the values with the ones you want to use.

```bash
CONFIG_FILE="configs/base_config.yaml" && \
EXPERIMENT_ID="20250905-170823-commit_05ef08d-config_196aafedf3" && \
CHECKPOINT_ID="checkpoint_foundational_epoch_1" && \
MODEL_PATH="experiments/${EXPERIMENT_ID}/${CHECKPOINT_ID}.pth" && \
PROMPT="the ocean reflects the" && \
MAX_TOKENS=50 && \
TEMPERATURE=0.8 && \
TOP_K=50
```

Then run the following command:

```bash
python3 Generator.py --config "${CONFIG_FILE}" --checkpoint "${MODEL_PATH}" --prompt "${PROMPT}" --max_tokens ${MAX_TOKENS} --temperature ${TEMPERATURE} --top_k ${TOP_K}
```

This can be done while the model trains. Throughout the training, the model checkpoints will be saved to the `experiments` directory.
