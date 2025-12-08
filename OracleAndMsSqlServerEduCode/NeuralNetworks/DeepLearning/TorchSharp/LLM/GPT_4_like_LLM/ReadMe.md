Access to the GPT-4-like code is restricted. 

**Please scroll down for the English version :-)** 

**CZ**

Nemá někdo (nejlépe z ČR, SR, PL, AT, HU, to abychom neměli k sobě daleko) chuť se přidat k mému side-projektu "mini LLMka" pro .NET v [tomto .NET jazyce](https://github.com/MiroslavHustak/FAQ) a TorchSharp pro lokální fine-tuning dat (pro .NET-only prostředí), která nemohou opustit "bezpečnostní prostor" uživatele? Nemohu slíbit, že to bude fungovat dle mých představ, ale pokud ano, můžeme uvažovat o společné monetizaci projektu. Bude to založeno na tomto [GPT-2-like" kódu](https://github.com/MiroslavHustak/ML_DL_SQL_Educational_Code/tree/master/OracleAndMsSqlServerEduCode/NeuralNetworks/DeepLearning/TorchSharp/LLM/GPT-2-like_LLM) (architekturou, ne výkonem) s přidanými prvky používaných v moderních modelech, už jsem do kódu v private repository přidal:
- rotary positional embeddings
- root mean square layer normalization (RMSNorm)
- LoRA
- batching
- přípravu pro learning rate scheduler

Vše bez optimalizace parametrů.

Zbývá ještě minimálně:
- flash attention a fuse ops (zřejmě bude nutné to udělat v C++, rád bych sice Rust, který se učím, ale nevím, jestli to v něm půjde, asi ne ...)
- implementace real-world tokenizátoru (zatím mám jen jednoduchou simulaci)
- implementace dostupných open weights
- kód pro zpracování dat pro fine-tuning

Konkurence samozřejmě existuje (např. Hugging Face Transformers + PEFT + vLLM nebo TensorRT-LLM, a hlavně pak Ollama + lokální fine-tuning či dnes už možná budou uvolněné lokální modely od OpenAI a dalších firem). U konkurence se dá očekávat, že bude příliš komplexní pro malé projekty – vlastní mini-framework umožní plnou kontrolu nad architekturou a experimenty. Bude to chtít společným úsilím sehnat přístup k PC s CUDA-compatible GPU. Prosím ozvěte se přes DM nebo email uvedený v kontaktech jen v případě vážného zájmu. Díky :-).

**EN**

Looking for a collaborator to join my side project: a “mini LLM” completely in .NET (and in F#).

You are ideally from the Czech Republic, Slovakia, Poland, Austria or Hungary – the closer the better; however if you're based somewhere else and your time zone isn’t too painful for a Central European, feel free to reach out too.

We’ll be using TorchSharp, with the goal of doing fully local fine-tuning of models whose data must never leave the user’s “secure environment”. I can’t promise it will turn out exactly as I imagine, but if it does, we can seriously think about joint monetisation. The foundation is my own from-scratch [GPT-2-like transformer](https://github.com/MiroslavHustak/ML_DL_SQL_Educational_Code/tree/master/OracleAndMsSqlServerEduCode/NeuralNetworks/DeepLearning/TorchSharp/LLM/GPT-2-like_LLM) (same architecture, obviously not the same performance). 

In my private repo I’ve already added:
- Rotary positional embeddings (RoPE)
- Root mean square layer normalization (RMSNorm)
- LoRA
- Proper batching
- Groundwork for LR schedulers

All of this still completely un-optimised. 

Still on the to-do list (quite a lot, hence the call for help):
- Flash Attention and kernel fusions (will probably have to be written in C++; I’d love to do it in Rust which I’m learning, but I’m not sure it’s realistic)
- Proper real-world tokenizer (right now there’s only a dummy placeholder)
- Loading existing open-weight models
- Full data-pre-processing pipeline for fine-tuning
- Other stuff I'm not aware yet

Yes, competition exists – Hugging Face Transformers + PEFT, vLLM, TensorRT-LLM, Ollama, maybe even local models from OpenAI & co. by the time we’re done.
The difference: those solutions are huge and heavyweight. Our own tiny framework will give us 100 % control over every line of code and make experimentation actually fun. We’ll need to figure out (together) how to get access to a decent CUDA GPU for development. If you’re seriously interested, drop me a DM or use the e-mail in my profile. Please spare me casual “sounds cool” replies – only genuine interest, thanks! :-)
